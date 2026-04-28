import React, { useState, useRef } from 'react';

const API = 'http://localhost:5002';

const ICONS = {
  recyclable: '♻️',
  organic:    '🌱',
  landfill:   '🗑️',
};

const COLORS = {
  recyclable: '#2196F3',
  organic:    '#4CAF50',
  landfill:   '#757575',
};

export default function App() {
  const [preview,  setPreview]  = useState(null);
  const [file,     setFile]     = useState(null);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFile = f => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop = e => {
    e.preventDefault(); setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const classify = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    const form = new FormData();
    form.append('image', file);
    try {
      const res = await fetch(`${API}/classify`, { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Classification failed. Make sure the Flask API is running.');
    }
    setLoading(false);
  };

  const accent = result ? COLORS[result.category] : '#333';

  return (
    <div style={{ fontFamily: 'system-ui', background: '#f5f6fa', minHeight: '100vh' }}>
      <header style={{ background: '#2d6a4f', color: '#fff', padding: '2rem', textAlign: 'center' }}>
        <h1 style={{ margin: 0 }}>♻️ Waste Classifier</h1>
        <p style={{ opacity: 0.75, margin: '8px 0 0' }}>
          Photograph an item to find out how to dispose of it
        </p>
      </header>

      <main style={{ maxWidth: 640, margin: '2rem auto', padding: '0 1rem', display: 'grid', gap: '1.5rem' }}>

        {/* Upload zone */}
        <div
          onClick={() => inputRef.current.click()}
          onDrop={onDrop}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          style={{
            border: `2px dashed ${dragging ? '#2d6a4f' : '#ccc'}`,
            borderRadius: 12, padding: '2rem', textAlign: 'center', cursor: 'pointer',
            background: dragging ? '#e8f5e9' : '#fff',
            transition: 'all 0.2s',
          }}>
          <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])} />
          {preview ? (
            <img src={preview} alt="preview"
              style={{ maxHeight: 280, maxWidth: '100%', borderRadius: 8, objectFit: 'contain' }} />
          ) : (
            <>
              <div style={{ fontSize: 48, marginBottom: 12 }}>📷</div>
              <p style={{ color: '#666' }}>Drag & drop a photo here, or click to browse</p>
            </>
          )}
        </div>

        {preview && (
          <button onClick={classify} disabled={loading}
            style={{ padding: '14px', background: '#2d6a4f', color: '#fff',
                     border: 'none', borderRadius: 8, fontSize: '1rem',
                     fontWeight: 600, cursor: 'pointer', opacity: loading ? 0.7 : 1 }}>
            {loading ? 'Classifying…' : 'Classify Item'}
          </button>
        )}

        {error && (
          <div style={{ background: '#fde', border: '1px solid #e74c3c', color: '#c0392b',
                         padding: '1rem', borderRadius: 8 }}>{error}</div>
        )}

        {result && (
          <div style={{ background: '#fff', borderRadius: 12, padding: '1.5rem',
                         boxShadow: '0 2px 12px rgba(0,0,0,0.1)', borderTop: `4px solid ${accent}` }}>
            <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
              <div style={{ fontSize: 56 }}>{ICONS[result.category]}</div>
              <h2 style={{ color: accent, textTransform: 'capitalize', margin: '8px 0 4px' }}>
                {result.category}
              </h2>
              <p style={{ color: '#666', margin: 0 }}>
                {(result.confidence * 100).toFixed(1)}% confidence
              </p>
            </div>

            <div style={{ background: '#f9f9f9', borderRadius: 8, padding: '1rem', marginBottom: '1.5rem' }}>
              <p style={{ margin: 0, lineHeight: 1.6 }}>{result.description}</p>
            </div>

            <h4 style={{ margin: '0 0 0.75rem' }}>All Category Scores</h4>
            {Object.entries(result.all_probs).map(([cat, prob]) => (
              <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <span style={{ width: 90, fontSize: 13, textTransform: 'capitalize' }}>{cat}</span>
                <div style={{ flex: 1, background: '#eee', borderRadius: 4, height: 10, overflow: 'hidden' }}>
                  <div style={{ width: `${prob * 100}%`, height: '100%',
                                 background: COLORS[cat], borderRadius: 4 }} />
                </div>
                <span style={{ width: 44, fontSize: 12, color: '#666', textAlign: 'right' }}>
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
