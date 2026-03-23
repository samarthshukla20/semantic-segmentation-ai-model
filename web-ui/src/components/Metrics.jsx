import { useEffect, useState, useRef } from 'react';
import { useScrollReveal } from '../hooks/useScrollReveal';
import './Metrics.css';

// UPGRADED: New Fine-Tuned EfficientNet Metrics (Sorted highest to lowest)
const IOU_DATA = [
    { name: 'Sky', value: 98.77, color: '#87CEEB' },
    { name: 'Trees', value: 88.59, color: '#228B22' },
    { name: 'Lush Bushes', value: 73.16, color: '#9ACD32' },
    { name: 'Landscape', value: 72.69, color: '#F4A460' },
    { name: 'Dry Grass', value: 71.88, color: '#DAA520' },
    { name: 'Flowers', value: 71.31, color: '#FF69B4' },
    { name: 'Logs', value: 65.82, color: '#A0522D' },
    { name: 'Rocks', value: 57.49, color: '#696969' },
    { name: 'Dry Bushes', value: 53.33, color: '#8B4513' },
    { name: 'Ground Clutter', value: 45.78, color: '#808080' },
];

function AnimatedNum({ to, visible, suffix = '' }) {
    const [val, setVal] = useState(0);
    const start = useRef(null);

    useEffect(() => {
        if (!visible) return;
        const run = (ts) => {
            if (!start.current) start.current = ts;
            const t = Math.min((ts - start.current) / 1800, 1);
            const ease = 1 - Math.pow(1 - t, 3);
            setVal(ease * to);
            if (t < 1) requestAnimationFrame(run);
        };
        requestAnimationFrame(run);
    }, [visible, to]);

    return <>{val.toFixed(val >= 10 ? 1 : 2)}{suffix}</>;
}

export default function Metrics() {
    const [headerRef, headerVisible] = useScrollReveal(0.2);
    const [barsRef, barsVisible] = useScrollReveal(0.1);

    return (
        <section id="metrics" className="section metrics-section">
            <div className="container">
                <div
                    ref={headerRef}
                    className={`metrics__header ${headerVisible ? 'is-visible' : ''}`}
                >
                    <p className="metrics__label">Results</p>
                    <h2 className="metrics__title">Validation performance</h2>
                    <p className="metrics__desc">Trained with hybrid CrossEntropy + Dice loss, evaluated on held-out validation split.</p>
                </div>

                <div className={`metrics__cards ${headerVisible ? 'is-visible' : ''}`}>
                    <div className="metrics__card">
                        <div className="metrics__card-val">
                            {/* UPGRADED: New Pixel Accuracy */}
                            <AnimatedNum to={89.15} visible={headerVisible} suffix="%" />
                        </div>
                        <div className="metrics__card-key">Pixel Accuracy</div>
                    </div>
                    <div className="metrics__card">
                        <div className="metrics__card-val">
                            {/* UPGRADED: New Mean IoU Score */}
                            <AnimatedNum to={70} visible={headerVisible} suffix="%" />
                        </div>
                        <div className="metrics__card-key">Mean IoU</div>
                    </div>
                    <div className="metrics__card">
                        <div className="metrics__card-val">U-Net</div>
                        {/* UPGRADED: New Brain Architecture */}
                        <div className="metrics__card-key">EfficientNet-B4, 512px</div>
                    </div>
                </div>

                <div ref={barsRef} className={`metrics__bars-section ${barsVisible ? 'is-visible' : ''}`}>
                    <h3 className="metrics__bars-heading">Per-class IoU</h3>
                    <div className="metrics__bars">
                        {IOU_DATA.map((item, i) => (
                            <div key={item.name} className="metrics__bar-row">
                                <span className="metrics__bar-name">
                                    <span className="metrics__bar-dot" style={{ background: item.color }} />
                                    {item.name}
                                </span>
                                <div className="metrics__bar-track">
                                    <div
                                        className="metrics__bar-fill"
                                        style={{
                                            width: barsVisible ? `${item.value}%` : '0%',
                                            background: item.color,
                                            transitionDelay: `${i * 60}ms`,
                                        }}
                                    />
                                </div>
                                <span className="metrics__bar-val">{item.value.toFixed(2)}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
}