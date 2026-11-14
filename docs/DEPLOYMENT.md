# RailSetu AI — Production & Deployment Guide
_Architecture, Containerization, and Runtime Configuration_

---

## 1. Runtime Architecture

RailSetu AI runs as a decoupled client-server architecture:
- **Backend API**: Python 3.11+ / FastAPI, served via Uvicorn (ASGI) with structured JSON logging and CORS middleware.
- **Frontend SPA**: React 19 + TypeScript + Vite + Tailwind CSS + Lucide icons, built into static assets served via Nginx or Cloudflare Pages.
- **Artifact Store**: Local filesystem (`backend/artifacts/` and `reports/v4/`) caching trained models (`.joblib`), metadata (`.json`), and pre-computed schedule results (`.csv`).

---

## 2. Environment Configuration

Create a `.env` file in the project root:

```ini
APP_NAME="RailSetu AI"
APP_VERSION="1.0.0"
APP_ENV="production"
DEBUG=false
CORS_ORIGINS="http://localhost:5173,http://localhost:3000,https://railsetu-ai.internal"
PORT=8000
HOST="0.0.0.0"
```

---

## 3. Running Locally

### Backend
```bash
source .venv/bin/activate
export PYTHONPATH=.
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify via health check:
```bash
curl http://localhost:8000/health
# {"status":"ok","app":"RailSetu AI","version":"1.0.0","environment":"production"}
```

### Frontend
```bash
cd frontend
npm run dev
# Running on http://localhost:5173
```

---

## 4. Container Deployment (Docker)

### Backend `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY data/processed ./data/processed
COPY reports ./reports

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Frontend `Dockerfile`
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 5. Production Health & Monitoring

- **Liveness Probe**: `GET /health` (returns 200 OK)
- **Readiness Probe**: `GET /api/dashboard/summary` (verifies artifact loading and dataframe parsing)
- **Metrics & Telemetry**: FastAPI structured request logging with latency in milliseconds.
