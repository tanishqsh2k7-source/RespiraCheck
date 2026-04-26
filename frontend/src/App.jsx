import React, { useState, useRef, useEffect } from "react";
import "./style.css";

// SVG Icons as functional components
const ActivityIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
);

const UploadCloudIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path><path d="M12 12v9"></path><path d="m16 16-4-4-4 4"></path></svg>
);

const XIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
);

const ShieldAlertIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
);

const ScanLineIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"></path><path d="M17 3h2a2 2 0 0 1 2 2v2"></path><path d="M21 17v2a2 2 0 0 1-2 2h-2"></path><path d="M7 21H5a2 2 0 0 1-2-2v-2"></path><line x1="7" y1="12" x2="17" y2="12"></line></svg>
);

const BotIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>
);

const SendIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
);

const AlertCircleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
);

const API_URL = "http://localhost:5000";

// Simple markdown parser for Gemini responses
const renderMarkdown = (text) => {
  if (!text) return null;
  
  // Split by paragraphs
  const paragraphs = text.split('\n\n');
  
  return paragraphs.map((para, i) => {
    // Check if it's a list
    if (para.includes('\n* ') || para.includes('\n- ') || para.startsWith('* ') || para.startsWith('- ')) {
      const items = para.split('\n').filter(item => item.trim());
      return (
        <ul key={i}>
          {items.map((item, j) => {
            // Remove bullet point
            let content = item.replace(/^[\*\-]\s+/, '');
            // Parse bold text **text**
            const parts = content.split(/(\*\*.*?\*\*)/g);
            return (
              <li key={j}>
                {parts.map((part, k) => {
                  if (part.startsWith('**') && part.endsWith('**')) {
                    return <strong key={k}>{part.slice(2, -2)}</strong>;
                  }
                  return part;
                })}
              </li>
            );
          })}
        </ul>
      );
    }
    
    // Regular paragraph, parse bold text
    const parts = para.split(/(\*\*.*?\*\*)/g);
    return (
      <p key={i}>
        {parts.map((part, k) => {
          if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={k}>{part.slice(2, -2)}</strong>;
          }
          return part;
        })}
      </p>
    );
  });
};

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);
  const chatEndRef = useRef(null);

  // Chat State
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);

  // Scroll to bottom of chat
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, isChatLoading]);

  // Initial bot message when results arrive
  useEffect(() => {
    if (result) {
      const initialMsg = `I've analyzed the chest X-ray. The AI model predicts **${result.prediction}** with a confidence of **${(result.confidence * 100).toFixed(1)}%**.\n\nI'm your AI Medical Assistant. Do you have any questions about this result or pneumonia in general?`;
      
      setChatHistory([
        { role: "assistant", text: initialMsg }
      ]);
    } else {
      setChatHistory([]);
    }
  }, [result]);

  const handleFile = (f) => {
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
    setChatHistory([]);
  };

  const handleClear = (e) => {
    e.stopPropagation();
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setChatHistory([]);
    if (inputRef.current) inputRef.current.value = "";
  };

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

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || isChatLoading || !result) return;

    const userMsg = chatInput.trim();
    setChatInput("");
    
    // Add user message to UI immediately
    const updatedHistory = [...chatHistory, { role: "user", text: userMsg }];
    setChatHistory(updatedHistory);
    setIsChatLoading(true);

    try {
      // Send the history (excluding the very first auto-generated message which might not be needed by Gemini or we can send it)
      // Actually, it's better to send the history so Gemini knows what was discussed.
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg,
          history: updatedHistory.slice(0, -1), // Everything except the message we just added
          prediction_context: {
            prediction: result.prediction,
            confidence: result.confidence
          }
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setChatHistory(prev => [...prev, { 
          role: "assistant", 
          text: `⚠️ Error: ${data.error || "Failed to get response"}` 
        }]);
      } else {
        setChatHistory(prev => [...prev, { 
          role: "assistant", 
          text: data.reply 
        }]);
      }
    } catch (err) {
      setChatHistory(prev => [...prev, { 
        role: "assistant", 
        text: "⚠️ Network error. Could not reach the chat server." 
      }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const isNormal = result?.prediction === "NORMAL";
  const confidencePct = result ? (result.confidence * 100).toFixed(1) : 0;

  return (
    <>
      <div className="bg-effects"></div>
      <div className="particles"></div>

      <div className="app-container">
        {/* Navbar */}
        <nav className="navbar">
          <div className="nav-brand">
            <div className="brand-icon">
              <ActivityIcon />
            </div>
            <span className="brand-text">RespiraCheck</span>
          </div>
        </nav>

        <main className="main-content">
          {/* Left Column: Scanner */}
          <section className={`scanner-section ${result ? 'has-results' : ''}`}>
            <div className="glass-card">
              <header className="card-header">
                <h1 className="card-title">Chest X-Ray Analysis</h1>
                <p className="card-subtitle">Upload a scan for AI-powered pneumonia detection</p>
              </header>

              {error && (
                <div className="error-alert">
                  <AlertCircleIcon />
                  <span>{error}</span>
                </div>
              )}

              {!preview ? (
                <div
                  className={`dropzone ${isDragging ? 'dragging' : ''}`}
                  onClick={() => inputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
                  }}
                >
                  <div className="dropzone-icon">
                    <UploadCloudIcon />
                  </div>
                  <h3 className="dropzone-text">Click or drag image to upload</h3>
                  <p className="dropzone-hint">Supports JPG, PNG up to 5MB</p>
                  <input
                    ref={inputRef}
                    type="file"
                    accept=".jpg,.jpeg,.png"
                    style={{ display: "none" }}
                    onChange={(e) => handleFile(e.target.files[0])}
                  />
                </div>
              ) : (
                <div className="preview-area">
                  <img src={preview} alt="X-ray preview" className="preview-image" />
                  
                  {loading && (
                    <div className="loading-overlay">
                      <div className="spinner"></div>
                      <p style={{ fontWeight: 500 }}>Analyzing Scan...</p>
                      <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Applying Grad-CAM++ algorithms</p>
                    </div>
                  )}

                  {!loading && !result && (
                    <div className="preview-overlay">
                      <span className="file-name">{file?.name}</span>
                      <button className="remove-btn" onClick={handleClear} title="Remove image">
                        <XIcon />
                      </button>
                    </div>
                  )}
                </div>
              )}

              {!result && (
                <button
                  className="btn-primary"
                  disabled={!file || loading}
                  onClick={handleSubmit}
                >
                  {loading ? "Processing..." : "Analyze Image"}
                </button>
              )}
              
              {result && (
                <button
                  className="btn-primary"
                  onClick={handleClear}
                  style={{ background: 'rgba(255,255,255,0.1)' }}
                >
                  Analyze New Image
                </button>
              )}

              {result && (
                <div className="results-section">
                  <div style={{ textAlign: "center" }}>
                    <div className={`status-badge ${isNormal ? 'status-normal' : 'status-pneumonia'}`}>
                      {isNormal ? "Normal" : "Pneumonia Detected"}
                    </div>
                  </div>

                  <div className="confidence-container">
                    <div className="confidence-header">
                      <span>AI Confidence Score</span>
                      <span className="confidence-value">{confidencePct}%</span>
                    </div>
                    <div className="confidence-track">
                      <div 
                        className={`confidence-fill ${isNormal ? 'normal' : 'pneumonia'}`}
                        style={{ width: `${confidencePct}%` }}
                      ></div>
                    </div>
                  </div>

                  {result.gradcam_image && (
                    <div className="gradcam-viewer">
                      <div className="gradcam-header">
                        <ScanLineIcon />
                        <span>Grad-CAM++ Explainability Map</span>
                      </div>
                      <img
                        src={`data:image/png;base64,${result.gradcam_image}`}
                        alt="Grad-CAM overlay"
                        className="gradcam-image"
                      />
                    </div>
                  )}

                  <div className="medical-disclaimer">
                    <ShieldAlertIcon className="disclaimer-icon" />
                    <p className="disclaimer-text">
                      <strong>AI Screening Tool:</strong> This system provides AI-assisted analysis and is not a substitute for professional medical diagnosis. Please consult a qualified healthcare provider for clinical decisions.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* Right Column: Chatbot (Only visible when results are present) */}
          {result && (
            <section className="results-section chat-section">
              <div className="chat-panel">
                <header className="chat-header">
                  <div className="ai-avatar">
                    <BotIcon />
                  </div>
                  <div>
                    <h2 className="chat-title">Medical AI Assistant</h2>
                    <div className="chat-status">
                      <div className="status-dot"></div>
                      Online • Gemini 1.5 Flash
                    </div>
                  </div>
                </header>

                <div className="chat-messages">
                  {chatHistory.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                      <div className="message-bubble">
                        {msg.role === 'user' ? msg.text : renderMarkdown(msg.text)}
                      </div>
                    </div>
                  ))}
                  
                  {isChatLoading && (
                    <div className="typing-indicator">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div className="chat-input-area">
                  <form onSubmit={handleChatSubmit} className="chat-form">
                    <input
                      type="text"
                      className="chat-input"
                      placeholder="Ask about the results or pneumonia..."
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      disabled={isChatLoading}
                    />
                    <button 
                      type="submit" 
                      className="send-btn" 
                      disabled={!chatInput.trim() || isChatLoading}
                    >
                      <SendIcon />
                    </button>
                  </form>
                </div>
              </div>
            </section>
          )}
        </main>
      </div>
    </>
  );
}
