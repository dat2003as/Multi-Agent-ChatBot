import re
import json
import time
from datetime import datetime
from typing import Optional, List

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from app.dependencies import redis_client, verify_api_key
from app.debug.tracer import get_logger

log = get_logger(__name__)
router = APIRouter(
    prefix="/shrimp-prices", 
    tags=["shrimp-prices"],
    dependencies=[Depends(verify_api_key)]
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

TEPBAC_URL = "https://tepbac.com/gia-thuy-san/gia/tom"
CACHE_KEY  = "chatbot:shrimp_prices:data"
CACHE_TTL  = 15 * 60  # 15 minutes

# ──────────────────────────────────────────────
# Schemas (Production Standard)
# ──────────────────────────────────────────────

class ShrimpPriceRecord(BaseModel):
    category: str = Field(..., example="Tôm sú")
    category_slug: str = Field(..., example="su")
    code: str = Field(..., example="SU20")
    name: str = Field(..., example="Tôm sú 20 con/kg")
    size: Optional[int] = Field(None, example=20)
    price: str = Field(..., example="240.000 đ/kg")
    price_value: Optional[int] = Field(None, example=240000)
    unit: str = Field(..., example="đ/kg")
    change: str = Field("", example="▲ 0.8%")
    change_percentage: Optional[float] = Field(None, example=0.8)
    last_updated: str = Field("", example="4 ngày trước")
    url: str = Field("", example="https://tepbac.com/...")

class ShrimpSummaryItem(BaseModel):
    name: str
    code: str
    price: str

class ShrimpSummary(BaseModel):
    updated_date: str
    total_count: int
    increased_count: int
    decreased_count: int
    stable_count: int
    average_change_percentage: float
    highest_price: Optional[ShrimpSummaryItem] = None
    lowest_price: Optional[ShrimpSummaryItem] = None

class ShrimpPriceResponse(BaseModel):
    status: str = "success"
    message: str = "Data fetched successfully"
    updated_date: str
    updated_at: str
    total_count: int
    data: List[ShrimpPriceRecord]

class HealthStatus(BaseModel):
    status: str
    has_data: bool
    total_count: int
    cache_ttl_seconds: int

# ──────────────────────────────────────────────
# Logic (Internal)
# ──────────────────────────────────────────────

def _slug(ten: str) -> str:
    t = ten.lower()
    if "sú" in t:  return "su"
    if "thẻ" in t: return "the"
    return "khac"

def _parse_table(table, danh_muc: str) -> List[dict]:
    records = []
    tbody = table.find("tbody")
    rows  = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        # Cột 0: Ảnh + mã + tên
        cell  = cols[0]
        a_tag = cell.find("a")
        href  = a_tag["href"] if a_tag else ""
        link  = ("https://tepbac.com" + href) if href.startswith("/") else href

        spans    = cell.find_all("span")
        ma       = ""
        ten_loai = ""
        for sp in spans:
            txt = sp.get_text(strip=True)
            if re.match(r"^[A-Z][A-Z0-9]{2,}$", txt):
                ma = txt
            elif txt and ma and not ten_loai:
                ten_loai = txt

        if not ma:
            raw = cell.get_text(" ", strip=True)
            tokens = raw.split()
            if tokens and re.match(r"^[A-Z][A-Z0-9]{2,}$", tokens[0]):
                ma = tokens[0]
                ten_loai = " ".join(tokens[1:])

        half = len(ten_loai) // 2
        if half and ten_loai[:half].strip() == ten_loai[half:].strip():
            ten_loai = ten_loai[:half].strip()

        # Cột 1: Size
        size_raw = cols[1].get_text(strip=True)
        size     = int(re.sub(r"[^\d]", "", size_raw)) if re.search(r"\d", size_raw) else None

        # Cột giá: ghép số + đơn vị
        gia_raw       = ""
        don_vi        = ""
        thay_doi_raw  = ""
        cap_nhat      = ""

        col_texts = [c.get_text(strip=True) for c in cols[2:]]
        gia_idx = None
        for i, txt in enumerate(col_texts):
            if re.search(r"\d{2,3}[.,]\d{3}", txt) or (txt.isdigit() and len(txt) >= 2):
                gia_idx = i
                break

        if gia_idx is not None:
            gia_raw = col_texts[gia_idx]
            if "kg" in gia_raw.lower() or "con" in gia_raw.lower():
                don_vi = "đ/kg" if "kg" in gia_raw.lower() else "đ/con"
                next_i = gia_idx + 1
            else:
                next_i = gia_idx + 1
                if next_i < len(col_texts) and re.search(r"đ/(kg|con)", col_texts[next_i], re.IGNORECASE):
                    don_vi  = col_texts[next_i]
                    gia_raw = gia_raw + " " + don_vi
                    next_i += 1
                else:
                    don_vi = "đ/kg"

            if next_i < len(col_texts):
                thay_doi_raw = col_texts[next_i]
            if next_i + 1 < len(col_texts):
                cap_nhat = col_texts[next_i + 1]

        gia_num = re.sub(r"[^\d]", "", re.split(r"[đĐ]", gia_raw)[0])
        gia_so  = int(gia_num) if gia_num else None

        if "kg" in gia_raw.lower() or "kg" in don_vi.lower():
            don_vi = "đ/kg"
        elif "con" in gia_raw.lower() or "con" in don_vi.lower():
            don_vi = "đ/con"

        m = re.search(r"([▲▼])\s*([\d.]+)%", thay_doi_raw)
        bien_dong_pct = None
        if m:
            bien_dong_pct = float(m.group(2)) * (1 if m.group(1) == "▲" else -1)

        if not ma and not ten_loai:
            continue

        records.append({
            "category":      danh_muc,
            "category_slug": _slug(danh_muc),
            "code":            ma,
            "name":      ten_loai,
            "size":          size,
            "price":           gia_raw.strip(),
            "price_value":        gia_so,
            "unit":        don_vi,
            "change":      thay_doi_raw,
            "change_percentage": bien_dong_pct,
            "last_updated":      cap_nhat,
            "url":          link,
        })

    return records

def fetch_and_parse() -> List[dict]:
    session = cffi_requests.Session()
    resp = session.get(
        TEPBAC_URL,
        impersonate="chrome124",
        headers={
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
            "Referer":         "https://tepbac.com/",
        },
        timeout=20,
    )
    if resp.status_code != 200:
        log.error("shrimp_prices.fetch_failed", status_code=resp.status_code)
        raise HTTPException(status_code=502, detail="tepbac.com không phản hồi. Vui lòng thử lại sau.")

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []
    seen_tables = set()

    for h2 in soup.find_all("h2"):
        ten_danh_muc = h2.get_text(strip=True)
        if not re.search(r"tôm", ten_danh_muc, re.IGNORECASE):
            continue

        table = h2.find_next("table")
        if table is None or id(table) in seen_tables:
            continue
        seen_tables.add(id(table))
        records += _parse_table(table, ten_danh_muc)

    return records

async def get_shrimp_data() -> List[dict]:
    if redis_client is None:
        # Fallback to direct fetch if Redis is down (should not happen in prod)
        log.warning("shrimp_prices.redis_unavailable")
        return fetch_and_parse()

    cached = await redis_client.get(CACHE_KEY)
    if cached:
        return json.loads(cached)

    log.info("shrimp_prices.cache_miss.fetching_fresh_data")
    data = fetch_and_parse()
    await redis_client.setex(CACHE_KEY, CACHE_TTL, json.dumps(data))
    return data

# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@router.get("", response_model=ShrimpPriceResponse)
async def get_shrimp_prices(
    category: Optional[str] = Query(None, description="su | the | khac"),
    code:       Optional[str] = Query(None, description="Mã loài, vd: THE50, SU20"),
):
    """Lấy danh sách giá tôm mới nhất từ tepbac.com."""
    records = await get_shrimp_data()
    
    result = records
    if category:
        result = [r for r in result if r["category_slug"] == category.lower()]
    if code:
        result = [r for r in result if r["code"].upper() == code.upper()]

    return ShrimpPriceResponse(
        updated_date=datetime.today().strftime("%d/%m/%Y"),
        updated_at=datetime.now().strftime("%H:%M:%S"),
        total_count=len(result),
        data=result
    )

@router.get("/summary", response_model=ShrimpSummary)
async def get_shrimp_summary():
    """Thống kê tổng quan giá tôm trong ngày."""
    records = await get_shrimp_data()
    if not records:
        raise HTTPException(503, "Hiện chưa có dữ liệu.")

    tang    = [r for r in records if r["change_percentage"] and r["change_percentage"] > 0]
    giam    = [r for r in records if r["change_percentage"] and r["change_percentage"] < 0]
    on_dinh = [r for r in records if not r["change_percentage"]]

    theo_kg   = [r for r in records if r["unit"] == "đ/kg" and r["price_value"]]
    cao_nhat  = max(theo_kg, key=lambda r: r["price_value"]) if theo_kg else None
    thap_nhat = min(theo_kg, key=lambda r: r["price_value"]) if theo_kg else None
    bd_list   = [r["change_percentage"] for r in records if r["change_percentage"] is not None]

    return ShrimpSummary(
        updated_date=datetime.today().strftime("%d/%m/%Y"),
        total_count=len(records),
        increased_count=len(tang),
        decreased_count=len(giam),
        stable_count=len(on_dinh),
        average_change_percentage=round(sum(bd_list) / len(bd_list), 2) if bd_list else 0,
        highest_price=ShrimpSummaryItem(name=cao_nhat["name"], code=cao_nhat["code"], price=cao_nhat["price"]) if cao_nhat else None,
        lowest_price=ShrimpSummaryItem(name=thap_nhat["name"], code=thap_nhat["code"], price=thap_nhat["price"]) if thap_nhat else None,
    )

@router.get("/health", response_model=HealthStatus)
async def health():
    """Kiểm tra trạng thái dịch vụ và cache."""
    data = []
    if redis_client:
        cached = await redis_client.get(CACHE_KEY)
        if cached:
            data = json.loads(cached)
    
    return HealthStatus(
        status="ok",
        has_data=len(data) > 0,
        total_count=len(data),
        cache_ttl_seconds=CACHE_TTL
    )
