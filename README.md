# Sentiment Analysis REST API

A production-ready REST API for text sentiment analysis built with **FastAPI**, **Pydantic v2**, and **Hugging Face Transformers**. Designed as a portfolio project showcasing modern Python backend architecture, clean separation of concerns, secure error handling, automated testing, and containerization.

---

## 🌟 Key Features

- **Multi-class Sentiment Classification**: Detects `positive`, `negative`, and `neutral` sentiments with a calibrated confidence score between 0.0 and 1.0.
- **Single-Load Lifespan Pattern**: The ML model is loaded once into memory during application startup using FastAPI's `lifespan` manager, avoiding prohibitive per-request initialization latencies.
- **Separation of Liveness & Readiness**: The `/health` endpoint distinguishes between the HTTP server accepting requests (`api_active`) and the ML model being ready for inference (`model_loaded`).
- **Secure Error Handling**: Centralized exception handlers sanitize client responses to prevent leakage of internal system paths, dependencies, or stack traces, while keeping full diagnostic details in server logs.
- **Configurable Architecture**: Backed by `pydantic-settings` to load and validate configurations from environment variables or `.env` files.
- **Comprehensive Automated Testing**: 100% endpoint test coverage with Pytest and HTTPX, including edge cases, blank text validations, length constraints, and degraded states.
- **Docker Ready**: Packaged with a single-stage `python:3.11-slim` container running under a dedicated non-root user.

---

## 🏗️ Project Architecture

```text
sentiment-analysis-api/
├── .env.example              # Template for environment configuration
├── .gitignore                # Git exclusions (caches, virtual environments, models)
├── .dockerignore             # Docker build context exclusions
├── Dockerfile                # Single-stage container based on python:3.11-slim
├── README.md                 # Project documentation and guide
├── requirements.txt          # Python dependencies
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI initialization, lifespan management, and router mounting
│   ├── config.py             # Pydantic Settings and environment configuration
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py     # Aggregated v1 endpoints router
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py # GET /health (distinguishes liveness vs readiness)
│   │           └── predict.py# POST /predict (text validation and inference)
│   ├── core/
│   │   ├── __init__.py
│   │   └── exceptions.py     # Centralized exception handlers preventing data leaks
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── health.py         # Response schema for /health
│   │   └── sentiment.py      # Request and response schemas with Pydantic validation
│   └── services/
│       ├── __init__.py
│       └── sentiment_model.py# Hugging Face Transformers pipeline wrapper and label mapping
└── tests/
    ├── __init__.py
    ├── conftest.py           # Pytest fixtures and mock service for fast execution
    ├── test_health.py        # Health endpoint tests (healthy & degraded)
    └── test_predict.py       # Predict endpoint tests (valid, blank, limits, 503)
```

---

## ⚙️ Configuration & Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | `str` | `development` | Deployment environment (`development`, `testing`, `production`) |
| `LOG_LEVEL` | `str` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `HOST` | `str` | `0.0.0.0` | Server bind host address |
| `PORT` | `int` | `8000` | Server bind port |
| `MODEL_NAME` | `str` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Hugging Face model identifier |
| `MAX_TEXT_LENGTH` | `int` | `500` | Maximum character length allowed for input text |

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.11 or higher
- Git

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/sentiment-analysis-api.git
cd sentiment-analysis-api
```

### 2. Create and activate a virtual environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 4. Configure environment
```bash
cp .env.example .env
```

### 5. Run the application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API documentation will be available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Endpoints & Usage

### 1. Health Check (`GET /health`)
Verifies service liveness and checks if the model is loaded into memory.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "api_active": true,
  "model_loaded": true,
  "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "environment": "development",
  "version": "1.0.0"
}
```

---

### 2. Predict Sentiment (`POST /predict`)
Analyzes the sentiment of a given text string.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The performance of this API is exceptional and super responsive!"}'
```

**Response (200 OK):**
```json
{
  "text": "The performance of this API is exceptional and super responsive!",
  "sentiment": "positive",
  "confidence": 0.985
}
```

#### Validation Error Handling (`422 Unprocessable Entity`)
If text is empty, contains only whitespace, or exceeds `MAX_TEXT_LENGTH`:

**Request (empty text):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "   "}'
```

**Sanitized Response (422):**
```json
{
  "detail": [
    {
      "loc": ["text"],
      "msg": "Value error, Text cannot be empty or contain only whitespace.",
      "type": "value_error"
    }
  ]
}
```

---

## 🧪 Running Automated Tests

Run the full Pytest suite:
```bash
pytest -v
```

All 12 test cases execute with injected mock services to run deterministically in milliseconds without relying on external network requests:
```text
tests/test_health.py::test_health_check_healthy PASSED                   [  8%]
tests/test_health.py::test_health_check_degraded PASSED                  [ 16%]
tests/test_health.py::test_health_check_v1_prefix PASSED                 [ 25%]
tests/test_predict.py::test_predict_positive_sentiment PASSED            [ 33%]
tests/test_predict.py::test_predict_negative_sentiment PASSED            [ 41%]
tests/test_predict.py::test_predict_neutral_sentiment PASSED             [ 50%]
tests/test_predict.py::test_predict_empty_text_returns_422 PASSED        [ 58%]
tests/test_predict.py::test_predict_whitespace_only_returns_422 PASSED   [ 66%]
tests/test_predict.py::test_predict_missing_text_field_returns_422 PASSED [ 75%]
tests/test_predict.py::test_predict_text_exceeding_max_length_returns_422 PASSED [ 83%]
tests/test_predict.py::test_predict_when_model_not_ready_returns_503 PASSED [ 91%]
tests/test_predict.py::test_predict_v1_alias PASSED                      [100%]
```

---

## 🐳 Docker Deployment

### Build the container
```bash
docker build -t sentiment-analysis-api:1.0.0 .
```

### Run the container
```bash
docker run -d \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  --name sentiment-api \
  sentiment-analysis-api:1.0.0
```

Verify container health:
```bash
docker ps
curl http://localhost:8000/health
```

---

## 🛡️ Security & Reliability Best Practices

1. **Information Leakage Prevention**: Stack traces and raw internal exceptions are logged on the server and never returned in client HTTP responses.
2. **Deterministic Startup**: Model failures during lifespan startup do not crash the container silently; the service transitions to a `degraded` state and responds with HTTP 503 on prediction routes.
3. **Principle of Least Privilege**: The Docker container executes under a dedicated non-root user (`appuser`).
4. **Clean Decoupling**: API routing, Pydantic schemas, ML inference logic, and configuration parameters are maintained in separate, testable modules.

---

## 📄 License
MIT License. Feel free to use and adapt for your own portfolio projects.
