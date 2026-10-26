# Sentiment Analysis API

A REST API for text sentiment classification built with FastAPI, Hugging Face Transformers, and PyTorch. The API classifies text as positive, negative, or neutral using the `cardiffnlp/twitter-roberta-base-sentiment-latest` model.

The application includes input validation, centralized error handling, automated tests, health monitoring, and Docker support.

## Features

- **Sentiment classification**: Predicts positive, negative, or neutral sentiment.
- **Confidence score**: Returns the model score associated with the predicted sentiment.
- **REST API**: FastAPI endpoints with interactive OpenAPI documentation.
- **Input validation**: Request and response schemas implemented with Pydantic.
- **Lifecycle management**: Loads the model during application startup.
- **Health monitoring**: Health endpoint and Docker health check.
- **Automated testing**: Unit tests and HTTP integration tests.
- **Containerization**: Docker image configured to run as a non-root user.
- **Persistent model cache**: Supports a Docker volume for Hugging Face model files.

## Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic and Pydantic Settings
- Hugging Face Transformers
- PyTorch
- Pytest, HTTPX, and pytest-asyncio
- Docker and Docker Compose-compatible container runtime

## Architecture

```text
Client
  │
  ▼
FastAPI application
  │
  ├──> Request validation (Pydantic)
  ├──> API routes
  │     │
  │     └──> Sentiment model service
  │           │
  │           └──> Hugging Face Transformers
  │                 │
  │                 └──> RoBERTa sentiment model
  ├──> Centralized exception handling
  └──> Health endpoint
```

The model is loaded during application startup and reused for subsequent inference requests.

## Requirements

- Python 3.11
- Docker Desktop or another compatible Docker Engine
- Internet access on first startup to download the model, unless it is already cached

## Run with Docker

### 1. Build the image

```bash
docker build -t sentiment-analysis-api:1.0.0 .
```

### 2. Start the container

```bash
docker run -d \
  --name sentiment-api \
  -p 8000:8000 \
  -v hf_model_cache:/app/.cache/huggingface \
  sentiment-analysis-api:1.0.0
```

The named volume preserves the Hugging Face model cache across container restarts and replacements.

### 3. Check the container

```bash
docker ps
docker logs -f sentiment-api
```

The first startup may take longer because the model must be downloaded and initialized.

### 4. Open the API documentation

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Checks application health and model readiness |
| `POST` | `/predict` | Predicts sentiment for a text input |

The application also exposes versioned API routes where configured by the API router.

### Health Check

**Request:**

```http
GET /health
```

Use this endpoint to inspect the application's health status. The Docker health check calls this route.

### Predict Sentiment

**Request:**

```http
POST /predict
Content-Type: application/json
```

**Body:**

```json
{
  "text": "The performance of this API is outstanding!"
}
```

**Example response:**

```json
{
  "text": "The performance of this API is outstanding!",
  "sentiment": "positive",
  "score": 0.9842
}
```

The score is the model's output for the predicted class. It should not be interpreted as a guarantee that the prediction is correct.

## Run Locally

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

Review `.env.example` and create a local `.env` file if required by your configuration. Do not commit secrets or environment-specific credentials.

### 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to interact with the API.

## Testing

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the standard test suite:

```bash
pytest -v -m "not integration"
```

Run the real-model integration tests separately:

```bash
pytest -v -m "integration"
```

Run the standard suite with coverage:

```bash
pytest -v -m "not integration" --cov=app --cov-report=term-missing
```

Real-model tests may require network access to download model artifacts on the first run and can take longer than unit tests.

## Docker Operations

- **View logs:**
  ```bash
  docker logs -f sentiment-api
  ```
- **Stop the container:**
  ```bash
  docker stop sentiment-api
  ```
- **Start it again:**
  ```bash
  docker start sentiment-api
  ```
- **Remove the container:**
  ```bash
  docker rm -f sentiment-api
  ```

The named model-cache volume is independent of the container and can be reused by a new container.

## Configuration

Application settings are managed through the project's configuration module and environment variables. Review `.env.example` and `app/config.py` for the exact supported settings, including the model name, environment, logging level, and CORS configuration.

## Security and Deployment Notes

- The Docker image runs the application as a non-root user.
- Do not commit `.env` files containing secrets.
- Configure CORS origins appropriately before exposing the API publicly.
- Use HTTPS and suitable access controls when deploying to a public environment.
- Consider resource limits and model inference latency when selecting a hosting platform.

## Current Status

- The Docker image has been built successfully, the container has started, the sentiment model has loaded, and the `/health` endpoint has returned HTTP 200.
- A prediction request was also tested successfully in the running container, returning positive with a score of 0.9842 for the example text.

## License

No license has been specified yet. Add a license file before distributing this project as open-source software.
