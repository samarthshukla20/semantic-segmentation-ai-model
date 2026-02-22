import { useEffect, useRef, useState } from 'react';
import { useParallax } from '../hooks/useScrollReveal';
import './Hero.css';

const ROTATING_WORDS = ['desert terrain', 'rocky paths', 'sandy dunes', 'offroad trails', 'rugged landscapes'];

export default function Hero() {
    const particlesRef = useRef(null);
    const [parallaxRef, parallaxOffset] = useParallax(0.25);
    const [wordIndex, setWordIndex] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);

    // Rotate words every 2.5s
    useEffect(() => {
        const interval = setInterval(() => {
            setIsAnimating(true);
            setTimeout(() => {
                setWordIndex(prev => (prev + 1) % ROTATING_WORDS.length);
                setIsAnimating(false);
            }, 550); // matches CSS transition duration
        }, 2500);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        const canvas = particlesRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let animationId;
        let particles = [];

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener('resize', resize);

        for (let i = 0; i < 45; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 1.8 + 0.3,
                speedX: Math.random() * 0.6 + 0.1,
                speedY: Math.random() * 0.2 - 0.1,
                opacity: Math.random() * 0.25 + 0.05,
            });
        }

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                p.x += p.speedX;
                p.y += p.speedY;
                if (p.x > canvas.width) { p.x = -5; p.y = Math.random() * canvas.height; }
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(180, 140, 90, ${p.opacity})`;
                ctx.fill();
            });
            animationId = requestAnimationFrame(animate);
        };
        animate();

        return () => {
            cancelAnimationFrame(animationId);
            window.removeEventListener('resize', resize);
        };
    }, []);

    return (
        <section id="hero" className="hero" ref={parallaxRef}>
            <canvas ref={particlesRef} className="hero__particles" />
            <div className="hero__bg" />
            <div className="hero__overlay" />

            <div
                className="hero__content container"
                style={{ transform: `translateY(${parallaxOffset}px)` }}
            >
                <p className="hero__kicker">Startathon 2025 — Offroad Segmentation</p>
                <h1 className="hero__title">
                    Mapping{' '}
                    <span className={`hero__rotating-word ${isAnimating ? 'exit' : 'enter'}`}>
                        {ROTATING_WORDS[wordIndex]}
                    </span>
                    <br />
                    <span className="hero__title-accent">for autonomous nav.</span>
                </h1>
                <p className="hero__subtitle">
                    A U-Net segmentation model that classifies 10 terrain types at 512px
                    resolution — trained on real offroad desert imagery.
                </p>
                <div className="hero__actions">
                    <a href="#try-it" className="hero__cta">
                        Run the model
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M12 5l7 7-7 7" /></svg>
                    </a>
                    <a href="#pipeline" className="hero__secondary">How it works</a>
                </div>

                <div className="hero__stats">
                    <div className="hero__stat">
                        <span className="hero__stat-value">87.8</span>
                        <span className="hero__stat-unit">%</span>
                        <span className="hero__stat-label">pixel accuracy</span>
                    </div>
                    <div className="hero__stat">
                        <span className="hero__stat-value">65.4</span>
                        <span className="hero__stat-unit">%</span>
                        <span className="hero__stat-label">mean IoU</span>
                    </div>
                    <div className="hero__stat">
                        <span className="hero__stat-value">10</span>
                        <span className="hero__stat-unit"></span>
                        <span className="hero__stat-label">terrain classes</span>
                    </div>
                    <div className="hero__stat">
                        <span className="hero__stat-value">512</span>
                        <span className="hero__stat-unit">px</span>
                        <span className="hero__stat-label">input res</span>
                    </div>
                </div>
            </div>
        </section>
    );
}
