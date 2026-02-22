import { useScrollReveal } from '../hooks/useScrollReveal';
import './Legend.css';

const CLASSES = [
    { name: 'Trees', color: '#228B22' },
    { name: 'Lush Bushes', color: '#9ACD32' },
    { name: 'Dry Grass', color: '#DAA520' },
    { name: 'Dry Bushes', color: '#8B4513' },
    { name: 'Ground Clutter', color: '#808080' },
    { name: 'Flowers', color: '#FF69B4' },
    { name: 'Logs', color: '#A0522D' },
    { name: 'Rocks', color: '#696969' },
    { name: 'Landscape', color: '#F4A460' },
    { name: 'Sky', color: '#87CEEB' },
];

export default function Legend() {
    const [headerRef, headerVisible] = useScrollReveal(0.2);
    const [gridRef, gridVisible] = useScrollReveal(0.1);

    return (
        <section id="legend" className="section legend-section">
            <div className="container">
                <div
                    ref={headerRef}
                    className={`legend__header ${headerVisible ? 'is-visible' : ''}`}
                >
                    <p className="legend__label">Classes</p>
                    <h2 className="legend__title">What the model sees</h2>
                    <p className="legend__desc">Each pixel is assigned to one of 10 terrain categories. Colors match the segmentation output.</p>
                </div>

                <div
                    ref={gridRef}
                    className={`legend__grid ${gridVisible ? 'is-visible' : ''}`}
                >
                    {CLASSES.map((cls, i) => (
                        <div
                            key={cls.name}
                            className="legend__item"
                            style={{ transitionDelay: `${i * 50}ms` }}
                        >
                            <span className="legend__swatch" style={{ background: cls.color }} />
                            <span className="legend__name">{cls.name}</span>
                            <span className="legend__hex">{cls.color}</span>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
