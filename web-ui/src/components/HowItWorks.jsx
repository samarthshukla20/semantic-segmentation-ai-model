import { useState } from 'react';
import { useScrollReveal, useParallax } from '../hooks/useScrollReveal';
import './HowItWorks.css';

const STEPS = [
    {
        num: '01',
        title: 'Upload',
        desc: 'Feed any offroad image into the pipeline — desert, rock, vegetation, anything.',
        detail: 'JPG / PNG, any resolution',
        icon: (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
        ),
    },
    {
        num: '02',
        title: 'Segment',
        desc: 'A U-Net with ResNet-34 backbone processes every pixel at 512×512.',
        detail: 'Hybrid CE + Dice loss',
        icon: (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" />
                <rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
        ),
    },
    {
        num: '03',
        title: 'Navigate',
        desc: 'Get a per-pixel classification across 10 terrain types, color-coded for readability.',
        detail: '~1s inference on CPU',
        icon: (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="3 11 22 2 13 21 11 13 3 11" />
            </svg>
        ),
    },
];

function PipelineCard({ step, index, active, onHover }) {
    const [ref, visible] = useScrollReveal(0.2);
    return (
        <div
            ref={ref}
            className={`pipeline__card ${visible ? 'is-visible' : ''} ${active ? 'is-active' : ''}`}
            style={{ transitionDelay: `${index * 120}ms` }}
            onMouseEnter={() => onHover(index)}
            onMouseLeave={() => onHover(-1)}
        >
            <div className="pipeline__card-icon">{step.icon}</div>
            <span className="pipeline__num">{step.num}</span>
            <h3 className="pipeline__card-title">{step.title}</h3>
            <p className="pipeline__card-desc">{step.desc}</p>
            <span className="pipeline__card-detail">{step.detail}</span>

            {/* Flow arrow between cards */}
            {index < STEPS.length - 1 && (
                <div className="pipeline__arrow">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="5" y1="12" x2="19" y2="12" />
                        <polyline points="12 5 19 12 12 19" />
                    </svg>
                </div>
            )}
        </div>
    );
}

export default function HowItWorks() {
    const [headerRef, headerVisible] = useScrollReveal(0.3);
    const [gridRef, gridOffset] = useParallax(0.08);
    const [activeCard, setActiveCard] = useState(-1);

    return (
        <section id="pipeline" className="section pipeline">
            <div className="container">
                <div
                    ref={headerRef}
                    className={`pipeline__header ${headerVisible ? 'is-visible' : ''}`}
                >
                    <p className="pipeline__label">Pipeline</p>
                    <h2 className="pipeline__title">Upload. Segment. Navigate.</h2>
                    <p className="pipeline__subtitle">Three steps from raw terrain image to navigable classification</p>
                </div>

                <div
                    ref={gridRef}
                    className="pipeline__grid"
                    style={{ transform: `translateY(${gridOffset}px)` }}
                >
                    {STEPS.map((step, i) => (
                        <PipelineCard
                            key={i}
                            step={step}
                            index={i}
                            active={activeCard === i}
                            onHover={setActiveCard}
                        />
                    ))}
                </div>

                {/* Progress dots */}
                <div className="pipeline__dots">
                    {STEPS.map((_, i) => (
                        <span
                            key={i}
                            className={`pipeline__dot ${activeCard === i ? 'is-active' : ''}`}
                        />
                    ))}
                </div>
            </div>
        </section>
    );
}
