import { useState, useEffect } from 'react';
import './Navbar.css';

const NAV_LINKS = [
    { id: 'hero', label: 'Home' },
    { id: 'pipeline', label: 'Pipeline' },
    { id: 'try-it', label: 'Demo' },
    { id: 'metrics', label: 'Results' },
    { id: 'legend', label: 'Classes' },
];

export default function Navbar() {
    const [scrolled, setScrolled] = useState(false);
    const [active, setActive] = useState('hero');

    useEffect(() => {
        const onScroll = () => {
            setScrolled(window.scrollY > 60);
            for (const link of [...NAV_LINKS].reverse()) {
                const el = document.getElementById(link.id);
                if (el && el.getBoundingClientRect().top <= 120) {
                    setActive(link.id);
                    break;
                }
            }
        };
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    return (
        <nav className={`navbar ${scrolled ? 'navbar--scrolled' : ''}`}>
            <div className="navbar__inner container">
                <a href="#hero" className="navbar__logo">
                    <svg className="navbar__logo-mark" width="28" height="28" viewBox="0 0 32 32" fill="none">
                        <path d="M4 26 Q10 14 16 18 Q22 22 28 10" stroke="var(--terracotta)" strokeWidth="2.5" strokeLinecap="round" fill="none" />
                        <path d="M2 28 L30 28" stroke="var(--sand-dark)" strokeWidth="1.5" strokeLinecap="round" opacity="0.4" />
                    </svg>
                    <span className="navbar__logo-text">desertnav</span>
                </a>
                <ul className="navbar__links">
                    {NAV_LINKS.map(link => (
                        <li key={link.id}>
                            <a
                                href={`#${link.id}`}
                                className={`navbar__link ${active === link.id ? 'navbar__link--active' : ''}`}
                            >
                                {link.label}
                            </a>
                        </li>
                    ))}
                </ul>
            </div>
        </nav>
    );
}
