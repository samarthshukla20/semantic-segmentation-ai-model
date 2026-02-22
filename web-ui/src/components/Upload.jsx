import { useState, useRef, useCallback } from 'react';
import { useScrollReveal } from '../hooks/useScrollReveal';
import './Upload.css';

const API_URL = 'http://localhost:5050/predict';

export default function Upload() {
    const [headerRef, headerVisible] = useScrollReveal(0.2);
    const [dragOver, setDragOver] = useState(false);
    const [image, setImage] = useState(null);
    const [imageUrl, setImageUrl] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [scanline, setScanline] = useState(false);
    const fileInputRef = useRef();

    const handleFile = useCallback((file) => {
        if (!file || !file.type.startsWith('image/')) return;
        setImage(file);
        setImageUrl(URL.createObjectURL(file));
        setResult(null);
        setError(null);
    }, []);

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        handleFile(e.dataTransfer.files[0]);
    };

    const handlePredict = async () => {
        if (!image) return;
        setLoading(true);
        setScanline(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('image', image);
            const res = await fetch(API_URL, { method: 'POST', body: formData });
            if (!res.ok) throw new Error('Prediction failed');
            const data = await res.json();
            setResult(data);
        } catch {
            setError(
                'Could not reach the API. Run: python web-ui/api_server.py'
            );
        } finally {
            setLoading(false);
            setTimeout(() => setScanline(false), 600);
        }
    };

    const reset = () => {
        setImage(null);
        setImageUrl(null);
        setResult(null);
        setError(null);
    };

    return (
        <section id="try-it" className="section upload-section">
            <div className="container">
                <div
                    ref={headerRef}
                    className={`upload__header ${headerVisible ? 'is-visible' : ''}`}
                >
                    <p className="upload__label">Demo</p>
                    <h2 className="upload__title">Run inference on your own image</h2>
                    <p className="upload__desc">Drop a terrain photo and see what the model outputs.</p>
                </div>

                <div className="upload__content">
                    {!imageUrl ? (
                        <div
                            className={`upload__dropzone ${dragOver ? 'upload__dropzone--active' : ''}`}
                            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                            onDragLeave={() => setDragOver(false)}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                onChange={(e) => handleFile(e.target.files[0])}
                                style={{ display: 'none' }}
                            />
                            <svg className="upload__dropzone-icon" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="17 8 12 3 7 8" />
                                <line x1="12" y1="3" x2="12" y2="15" />
                            </svg>
                            <p className="upload__dropzone-text">
                                Drop image here or <strong>browse</strong>
                            </p>
                            <span className="upload__dropzone-hint">JPG, PNG — any resolution</span>
                        </div>
                    ) : (
                        <div className="upload__results">
                            <div className="upload__images-row">
                                <div className="upload__image-card">
                                    <span className="upload__image-label">Input</span>
                                    <div className="upload__image-wrapper">
                                        <img src={imageUrl} alt="Uploaded terrain" />
                                    </div>
                                </div>

                                <div className="upload__image-card">
                                    <span className="upload__image-label">Prediction</span>
                                    <div className={`upload__image-wrapper ${scanline ? 'upload__image-wrapper--scanning' : ''}`}>
                                        {result ? (
                                            <img src={`data:image/png;base64,${result.mask}`} alt="Segmentation" />
                                        ) : (
                                            <div className="upload__image-placeholder">
                                                {loading ? (
                                                    <div className="upload__spinner" />
                                                ) : (
                                                    <span>Hit "Run" to segment</span>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {result && result.confidence && (
                                <div className="upload__meta">
                                    <div className="upload__confidence">
                                        <span className="upload__confidence-label">Confidence</span>
                                        <div className="upload__confidence-bar-track">
                                            <div
                                                className="upload__confidence-bar-fill"
                                                style={{ width: `${result.confidence}%` }}
                                            />
                                        </div>
                                        <span className="upload__confidence-value">{result.confidence.toFixed(1)}%</span>
                                    </div>
                                </div>
                            )}

                            {result && result.classes_detected && (
                                <div className="upload__breakdown">
                                    <span className="upload__breakdown-title">Composition</span>
                                    <div className="upload__breakdown-list">
                                        {result.classes_detected.map((cls) => (
                                            <div key={cls.name} className="upload__breakdown-row">
                                                <span className="upload__breakdown-dot" style={{ background: cls.color }} />
                                                <span className="upload__breakdown-name">{cls.name}</span>
                                                <div className="upload__breakdown-bar-track">
                                                    <div
                                                        className="upload__breakdown-bar-fill"
                                                        style={{ width: `${cls.percentage}%`, background: cls.color }}
                                                    />
                                                </div>
                                                <span className="upload__breakdown-pct">{cls.percentage}%</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {error && <div className="upload__error">{error}</div>}

                            <div className="upload__actions">
                                <button className="upload__run-btn" onClick={handlePredict} disabled={loading}>
                                    {loading ? 'Running...' : 'Run model'}
                                </button>
                                <button className="upload__reset-btn" onClick={reset}>
                                    Reset
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
}
