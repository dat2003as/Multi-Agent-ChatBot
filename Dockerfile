FROM python:3.11-slim

WORKDIR /app

# Env cho dev (log rõ, không cache pyc)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài dependencies cần thiết (dev thì cứ giữ build tools cho tiện)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng cache
COPY requirements.txt .

# Install python packages
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Tạo thư mục logs
RUN mkdir -p logs

# Expose port
EXPOSE 1012

# Dev thì nên bật reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1012", "--reload"]