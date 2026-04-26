/**
 * App.jsx — PneumoScan React Frontend
 *
 * Single-file React component providing a medical-style dark-themed UI
 * for uploading chest X-ray images, obtaining AI predictions, and
 * visualising Grad-CAM++ explainability overlays.
 *
 * Communicates with the Flask backend at http://localhost:5000.
 */

import React, { useState, useRef, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Configuration                                                      */
/* ------------------------------------------------------------------ */
const API_URL = "http://localhost:5000";

/* ------------------------------------------------------------------ */
/*  Styles                                                             */
/* ------------------------------------------------------------------ */
const styles = {
  /* ---------- Global / Layout ---------- */
  app: {
    minHeight: "100vh",
    background: "linear-gradient(145deg, #0a0f1e 0%, #0d1528 50%, #0a1020 100%)",
    color: "#e0e6f0",
    fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "0 16px 48px",
  },

  /* ---------- Header ---------- */
  header: {
    width: "100%",
    maxWidth: 860,
    textAlign: "center",
    padding: "40px 0 24px",
  },
  title: {
    fontSize: 32,
    fontWeight: 700,
    background: "linear-gradient(90deg, #00d4ff, #7b61ff)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    margin: 0,
  },
  subtitle: {
    fontSize: 14,
    color: "#8892a7",
    marginTop: 6,
    letterSpacing: 0.3,
  },

  /* ---------- Card container ---------- */
  card: {
    width: "100%",
    maxWidth: 620,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: "32px 28px",
    backdropFilter: "blur(12px)",
    boxShadow: "0 8px 32px rgba(0,0,0,0.35)",
    marginTop: 8,
  },

  /* ---------- Drop zone ---------- */
  dropZone: (isDragging) => ({
    border: `2px dashed ${isDragging ? "#00d4ff" : "rgba(255,255,255,0.15)"}`,
    borderRadius: 12,
    padding: "40px 20px",
    textAlign: "center",
    cursor: "pointer",
    transition: "all 0.25s ease",
    background: isDragging
      ? "rgba(0,212,255,0.06)"
      : "rgba(255,255,255,0.02)",
  }),
  dropIcon: {
    fontSize: 42,
    marginBottom: 10,
    filter: "grayscale(0.3)",
  },
  dropText: {
    fontSize: 15,
    color: "#a0acc4",
  },
  dropHint: {
    fontSize: 12,
    color: "#5e6a80",
    marginTop: 6,
  },

  /* ---------- Preview ---------- */
  previewContainer: {
    marginTop: 20,
    textAlign: "center",
  },
  previewImage: {
    maxWidth: "100%",
    maxHeight: 260,
    borderRadius: 10,
    border: "1px solid rgba(255,255,255,0.1)",
    objectFit: "contain",
  },
  fileName: {
    fontSize: 13,
    color: "#8892a7",
    marginTop: 8,
  },

  /* ---------- Button ---------- */
  button: (disabled) => ({
    marginTop: 22,
    width: "100%",
    padding: "14px 0",
    fontSize: 16,
    fontWeight: 600,
    letterSpacing: 0.4,
    color: disabled ? "#5e6a80" : "#0a0f1e",
    background: disabled
      ? "rgba(255,255,255,0.06)"
      : "linear-gradient(135deg, #00d4ff, #7b61ff)",
    border: "none",
    borderRadius: 10,
    cursor: disabled ? "not-allowed" : "pointer",
    transition: "all 0.3s ease",
    boxShadow: disabled ? "none" : "0 4px 18px rgba(0,212,255,0.25)",
  }),

  /* ---------- Spinner ---------- */
  spinnerWrap: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "36px 0",
  },
  spinnerText: {
    marginTop: 14,
    fontSize: 15,
    color: "#00d4ff",
    letterSpacing: 0.5,
  },

  /* ---------- Results ---------- */
  resultSection: {
    marginTop: 28,
  },
  badge: (isNormal) => ({
    display: "inline-block",
    padding: "8px 22px",
    borderRadius: 8,
    fontSize: 18,
    fontWeight: 700,
    letterSpacing: 1,
    color: "#fff",
    background: isNormal
      ? "linear-gradient(135deg, #00c853, #00e676)"
      : "linear-gradient(135deg, #ff1744, #ff5252)",
    boxShadow: isNormal
      ? "0 4px 14px rgba(0,200,83,0.3)"
      : "0 4px 14px rgba(255,23,68,0.3)",
  }),
  confidenceWrap: {
    marginTop: 18,
  },
  confidenceLabel: {
    fontSize: 13,
    color: "#8892a7",
    marginBottom: 6,
  },
  progressBarOuter: {
    width: "100%",
    height: 10,
    background: "rgba(255,255,255,0.08)",
    borderRadius: 5,
    overflow: "hidden",
  },
  progressBarInner: (pct, isNormal) => ({
    width: `${pct}%`,
    height: "100%",
    borderRadius: 5,
    background: isNormal
      ? "linear-gradient(90deg, #00c853, #69f0ae)"
      : "linear-gradient(90deg, #ff1744, #ff8a80)",
    transition: "width 0.8s ease",
  }),
  confidenceValue: {
    fontSize: 14,
    fontWeight: 600,
    color: "#e0e6f0",
    marginTop: 6,
  },

  /* ---------- Grad-CAM ---------- */
  gradcamWrap: {
    marginTop: 24,
    textAlign: "center",
  },
  gradcamTitle: {
    fontSize: 14,
    color: "#8892a7",
    marginBottom: 10,
  },
  gradcamImage: {
    maxWidth: "100%",
    maxHeight: 280,
    borderRadius: 10,
    border: "1px solid rgba(0,212,255,0.2)",
    objectFit: "contain",
  },

  /* ---------- Disclaimer ---------- */
  disclaimer: {
    marginTop: 24,
    padding: "12px 16px",
    background: "rgba(255,193,7,0.08)",
    border: "1px solid rgba(255,193,7,0.2)",
    borderRadius: 8,
    fontSize: 12,
    color: "#ffd54f",
    lineHeight: 1.5,
    textAlign: "center",
  },

  /* ---------- Error ---------- */
  errorBox: {
    marginTop: 20,
    padding: "12px 16px",
    background: "rgba(255,23,68,0.1)",
    border: "1px solid rgba(255,23,68,0.25)",
    borderRadius: 8,
    fontSize: 14,
    color: "#ff8a80",
    textAlign: "center",
  },
};

/* ------------------------------------------------------------------ */
/*  Spinner keyframes (injected once)                                  */
/* ------------------------------------------------------------------ */
const spinnerCSS = `
@keyframes pneumo-spin {
  0%   { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.pneumo-spinner {
  width: 44px;
  height: 44px;
  border: 4px solid rgba(255,255,255,0.1);
  border-top-color: #00d4ff;
  border-radius: 50%;
  animation: pneumo-spin 0.8s linear infinite;
}
`;

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  /* ---------- File handling ---------- */
  const handleFile = useCallback((f) => {
    if (!f) return;
    const ext = f.name.split(".").pop().toLowerCase();
    if (!["jpg", "jpeg", "png"].includes(ext)) {
      setError("Invalid file type. Please upload a JPG or PNG image.");
      return;
    }
    if (f.size > 5 * 1024 * 1024) {
      setError("File too large. Maximum size is 5 MB.");
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const onFileChange = (e) => handleFile(e.target.files[0]);

  /* ---------- Drag & Drop ---------- */
  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  };

  /* ---------- Analyze ---------- */
  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.error || "An unknown error occurred.");
      } else {
        setResult(data);
      }
    } catch (err) {
      setError("Failed to connect to server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  /* ---------- Render ---------- */
  const isNormal = result?.prediction === "NORMAL";
  const confidencePct = result ? (result.confidence * 100).toFixed(1) : 0;

  return (
    <>
      {/* Inject spinner animation */}
      <style>{spinnerCSS}</style>

      <div style={styles.app}>
        {/* Header */}
        <header style={styles.header}>
          <h1 style={styles.title}>PneumoScan</h1>
          <p style={styles.subtitle}>
            AI-Assisted Chest X-Ray Analysis
          </p>
        </header>

        {/* Main card */}
        <div style={styles.card}>
          {/* Upload zone */}
          <div
            style={styles.dropZone(isDragging)}
            onClick={() => inputRef.current?.click()}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            role="button"
            tabIndex={0}
            aria-label="Upload chest X-ray image"
          >
            <div style={styles.dropIcon}>🫁</div>
            <p style={styles.dropText}>
              Drag &amp; drop a chest X-ray here, or click to browse
            </p>
            <p style={styles.dropHint}>JPG / PNG — max 5 MB</p>
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              style={{ display: "none" }}
              onChange={onFileChange}
            />
          </div>

          {/* Preview */}
          {preview && (
            <div style={styles.previewContainer}>
              <img
                src={preview}
                alt="X-ray preview"
                style={styles.previewImage}
              />
              <p style={styles.fileName}>{file?.name}</p>
            </div>
          )}

          {/* Submit */}
          <button
            style={styles.button(!file || loading)}
            disabled={!file || loading}
            onClick={handleSubmit}
          >
            {loading ? "Analyzing…" : "Analyze X-Ray"}
          </button>

          {/* Loading */}
          {loading && (
            <div style={styles.spinnerWrap}>
              <div className="pneumo-spinner" />
              <p style={styles.spinnerText}>Analyzing…</p>
            </div>
          )}

          {/* Error */}
          {error && <div style={styles.errorBox}>{error}</div>}

          {/* Results */}
          {result && (
            <div style={styles.resultSection}>
              {/* Badge */}
              <div style={{ textAlign: "center" }}>
                <span style={styles.badge(isNormal)}>
                  {result.prediction}
                </span>
              </div>

              {/* Confidence */}
              <div style={styles.confidenceWrap}>
                <p style={styles.confidenceLabel}>Confidence</p>
                <div style={styles.progressBarOuter}>
                  <div
                    style={styles.progressBarInner(confidencePct, isNormal)}
                  />
                </div>
                <p style={styles.confidenceValue}>{confidencePct}%</p>
              </div>

              {/* Grad-CAM */}
              {result.gradcam_image && (
                <div style={styles.gradcamWrap}>
                  <p style={styles.gradcamTitle}>
                    Grad-CAM++ Explainability Overlay
                  </p>
                  <img
                    src={`data:image/png;base64,${result.gradcam_image}`}
                    alt="Grad-CAM overlay"
                    style={styles.gradcamImage}
                  />
                </div>
              )}

              {/* Disclaimer */}
              <div style={styles.disclaimer}>
                ⚠️ This is an AI screening tool. Results must be verified by a
                qualified radiologist. Do not use for clinical decision-making
                without professional oversight.
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
