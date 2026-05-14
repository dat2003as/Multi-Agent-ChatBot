"""
OpenAI function definitions for GPT-4.1-mini tool selection.

Tools:
  - search_documents   : RAG internal docs (SOP, FAQ, guide)
  - answer_general     : Aquaculture expert advice / chitchat
  - query_data         : Multi-domain SQL query (A1/A2/A3/A4 via Vanna)
  - search_realtime_info : Shrimp prices (tepbac.com) & weather (wttr.in)
"""
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search internal knowledge base: SOPs, app guides, "
                "technical FAQs, disease treatment protocols, equipment manuals. "
                "Use for how-to and procedure questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query in Vietnamese.",
                    },
                    "source_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["sop", "faq", "guide", "manual"]},
                        "description": "Optional: filter by document type.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_general",
            "description": (
                "Answer general aquaculture knowledge: disease symptoms, water quality, "
                "nutrition, pond management, or casual conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to answer.",
                    },
                    "subtopic": {
                        "type": "string",
                        "enum": ["disease_treatment", "water_quality", "feeding", "chitchat", "general_advisory"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "Query structured data from the farm database across 4 domains: "
                "A1 (farm operations), A2 (inventory), A3 (analytics), A4 (IoT devices). "
                "Use for: pond info, stock levels, KPIs, device status, harvest data, costs, equipment lists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question in Vietnamese.",
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["a1", "a2", "a3", "a4"],
                        "description": (
                            "a1: Farm ops — ponds, diseases, harvest, sensors. "
                            "a2: Inventory — stock, receipts, suppliers, material prices, material catalog. "
                            "a3: Analytics — farm-level KPIs, total costs, revenue, profit (NOT individual material prices). "
                            "a4: IoT — devices, aerators, pumps, cameras, sensors, cabinets, scales."
                        ),
                    },
                    "subtopic": {
                        "type": "string",
                        "enum": [
                            "pond_info", "operation_record", "disease_history",
                            "sensor_data", "harvest_yield",
                            "current_stock", "import_receipt", "export_receipt",
                            "inventory_check", "low_stock_alert", "material_catalog",
                            "supplier", "daily_report", "warehouse_lot", "central_warehouse",
                            "kpi_report", "cost_analysis", "revenue_profit", "farm_performance",
                            "device_list", "sensor_reading", "device_status",
                            "weight_data", "iot_monitoring", "device_history",
                        ],
                        "description": "Optional: subtopic to optimize the query.",
                    },
                },
                "required": ["query", "domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_realtime_info",
            "description": (
                "Look up real-time shrimp prices or weather forecast. "
                "For weather, 'location' is REQUIRED — if user doesn't specify, ask first, do NOT call this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's original question in Vietnamese.",
                    },
                    "info_type": {
                        "type": "string",
                        "enum": ["shrimp_prices", "weather"],
                        "description": "shrimp_prices: national market prices. weather: forecast for a location.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Required for weather. Vietnamese province/city name (e.g. 'Ca Mau', 'Soc Trang').",
                    },
                },
                "required": ["query", "info_type"],
            },
        },
    },
]
