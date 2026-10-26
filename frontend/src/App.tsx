
import { useState } from "react";
import "./App.css";

type SentimentResponse = {
  text: string;
  sentiment: string;
  score: number;
};

const API_URL = (
  import.meta.env.VITE_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<SentimentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyzeSentiment() {
    if (!text.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!response.ok) {
        throw new Error(
          response.status === 422
            ? "El texto no cumple las validaciones de la API."
            : `Error de la API: ${response.status}`
        );
      }

      const data: SentimentResponse = await response.json();
      setResult(data);
    } catch {
      setError(
        "No se pudo conectar con la API. Comprueba que Docker esté ejecutándose y que CORS esté configurado."
      );
    } finally {
      setLoading(false);
    }
  }

  const labels: Record<string, string> = {
    positive: "Positivo",
    negative: "Negativo",
    neutral: "Neutral",
  };

  const percentage = result
    ? Math.max(0, Math.min(100, result.score * 100))
    : 0;

  return (
    <main className="page">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-icon">S</span>
          <span>
            Sentiment<span className="brand-light">Analyzer</span>
          </span>
        </a>

        <span className="api-status">
          <span className="status-dot" />
          API · localhost:8000
        </span>
      </header>

      <section className="workspace">
        <div className="intro">
          <span className="eyebrow">ANÁLISIS DE TEXTO CON IA</span>
          <h1>
            Entiende el sentimiento
            <br />
            detrás de cada texto.
          </h1>
          <p>
            Analiza opiniones, reseñas y comentarios con tu modelo de
            inteligencia artificial.
          </p>
        </div>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Analizar texto</h2>
              <p>Introduce el texto que deseas evaluar.</p>
            </div>
            <span className="model-tag">RoBERTa</span>
          </div>

          <label className="field-label" htmlFor="sentiment-text">
            Tu texto
          </label>

          <textarea
            id="sentiment-text"
            value={text}
            onChange={(event) => setText(event.target.value.slice(0, 500))}
            placeholder="Ej.: I love this product. It works perfectly!"
            maxLength={500}
            rows={6}
          />

          <div className="input-footer">
            <span>{text.length}/500 caracteres</span>
            <button
              className="clear-button"
              type="button"
              onClick={() => {
                setText("");
                setResult(null);
                setError("");
              }}
            >
              Limpiar texto
            </button>
          </div>

          <button
            className="analyze-button"
            onClick={analyzeSentiment}
            disabled={loading || !text.trim()}
            type="button"
          >
            {loading ? "Analizando..." : "✳  Analizar sentimiento"}
          </button>

          {error && (
            <div className="error-message" role="alert">
              {error}
            </div>
          )}
        </section>

        {result && (
          <section className="result-panel" aria-live="polite">
            <div className="result-top">
              <div>
                <span className="eyebrow">RESULTADO DEL MODELO</span>
                <h2>Sentimiento detectado</h2>
              </div>

              <span className={`sentiment-badge ${result.sentiment}`}>
                <span className="result-dot" />
                {labels[result.sentiment] ?? result.sentiment}
              </span>
            </div>

            <div className="score-heading">
              <span>Puntuación del modelo</span>
              <strong>{percentage.toFixed(2)}%</strong>
            </div>

            <div className="score-track">
              <div
                className={`score-fill ${result.sentiment}`}
                style={{ width: `${percentage}%` }}
              />
            </div>

            <div className="analyzed-text">
              <span>Texto analizado</span>
              <p>{result.text}</p>
            </div>

            <p className="disclaimer">
              La puntuación es una salida del modelo, no una garantía de
              clasificación correcta.
            </p>
          </section>
        )}

        <footer>Desarrollado con React y FastAPI</footer>
      </section>
    </main>
  );
}

export default App;
