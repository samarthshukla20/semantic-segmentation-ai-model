import { useEffect, useState, useRef } from 'react';
import './ScrollScene.css';

/*
  Speed zones — the jeep slows through rocky/obstacle sections
  and speeds up on open sand. Each zone defines a scroll-progress
  range and relative speed multiplier.
*/
const SPEED_ZONES = [
    { start: 0.00, end: 0.12, speed: 1.0, label: 'open sand' },
    { start: 0.12, end: 0.22, speed: 0.4, label: 'rocky pass' },
    { start: 0.22, end: 0.40, speed: 1.2, label: 'open sand' },
    { start: 0.40, end: 0.52, speed: 0.35, label: 'boulder field' },
    { start: 0.52, end: 0.70, speed: 1.1, label: 'open sand' },
    { start: 0.70, end: 0.82, speed: 0.45, label: 'rough terrain' },
    { start: 0.82, end: 1.00, speed: 1.3, label: 'open sand' },
];

/* Build a non-linear position map from the speed zones */
function buildPositionMap() {
    let totalWeighted = 0;
    for (const z of SPEED_ZONES) {
        totalWeighted += (z.end - z.start) * z.speed;
    }

    // returns vehicle position (0→1) for a given scroll progress (0→1)
    return (progress) => {
        let accumulated = 0;
        for (const z of SPEED_ZONES) {
            if (progress <= z.start) break;
            const segEnd = Math.min(progress, z.end);
            accumulated += (segEnd - z.start) * z.speed;
            if (progress <= z.end) break;
        }
        return accumulated / totalWeighted;
    };
}

const getVehiclePosition = buildPositionMap();

/* Current zone for tilt/bounce behaviour */
function getCurrentZone(progress) {
    for (const z of SPEED_ZONES) {
        if (progress >= z.start && progress < z.end) return z;
    }
    return SPEED_ZONES[SPEED_ZONES.length - 1];
}

export default function ScrollScene() {
    const [scroll, setScroll] = useState(0);
    const rafRef = useRef(null);
    const scrollRef = useRef(0);

    useEffect(() => {
        const onScroll = () => { scrollRef.current = window.scrollY; };
        window.addEventListener('scroll', onScroll, { passive: true });

        const tick = () => {
            setScroll(prev => {
                const diff = scrollRef.current - prev;
                return prev + diff * 0.12;
            });
            rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);

        return () => {
            window.removeEventListener('scroll', onScroll);
            cancelAnimationFrame(rafRef.current);
        };
    }, []);

    const maxScroll = typeof document !== 'undefined'
        ? document.documentElement.scrollHeight - window.innerHeight
        : 3000;
    const progress = Math.min(scroll / Math.max(maxScroll, 1), 1);

    // Non-linear vehicle position
    const vehiclePos = getVehiclePosition(progress);
    const vehicleX = -5 + vehiclePos * 105;

    // Bounce & tilt based on current zone
    const zone = getCurrentZone(progress);
    const isRough = zone.speed < 0.6;
    const bounce = isRough
        ? Math.sin(progress * Math.PI * 40) * 4   // aggressive bounce on rough
        : Math.sin(progress * Math.PI * 12) * 1.5; // gentle on open sand
    const tilt = isRough
        ? Math.sin(progress * Math.PI * 28) * 3    // wobble on rocks
        : 0;

    // Dust intensity — more on open sand (faster), less on rocks (slower)
    const dustOpacity = isRough ? 0.04 : 0.12;

    // Terrain layers
    const dunesFar = progress * -120;
    const dunesMid = progress * -200;
    const dunesNear = progress * -320;
    const rocksShift = progress * -180;
    const cactiShift = progress * -250;

    return (
        <div className="scroll-scene" aria-hidden="true">

            {/* ── Far mountains ── */}
            <svg
                className="scroll-scene__layer scroll-scene__mountains"
                viewBox="0 0 1440 200"
                preserveAspectRatio="none"
                style={{ transform: `translateX(${dunesFar}px)` }}
            >
                <path d="M0 200 L80 120 L160 160 L280 80 L400 140 L520 60 L640 130 L760 50 L880 110 L1000 70 L1120 130 L1240 90 L1360 150 L1440 100 L1600 80 L1760 140 L1920 60 L2000 200Z"
                    fill="rgba(140, 110, 70, 0.4)" />
            </svg>

            {/* ── Mid dunes ── */}
            <svg
                className="scroll-scene__layer scroll-scene__dunes-mid"
                viewBox="0 0 1440 140"
                preserveAspectRatio="none"
                style={{ transform: `translateX(${dunesMid}px)` }}
            >
                <path d="M0 140 Q120 60 240 100 Q360 40 480 90 Q600 30 720 80 Q840 20 960 75 Q1080 35 1200 85 Q1320 50 1440 70 L1600 50 Q1720 30 1840 80 L2000 140Z"
                    fill="rgba(165, 135, 85, 0.3)" />
            </svg>

            {/* ── Obstacle rocks ── */}
            <svg
                className="scroll-scene__layer scroll-scene__rocks"
                viewBox="0 0 1440 80"
                preserveAspectRatio="none"
                style={{ transform: `translateX(${rocksShift}px)` }}
            >
                {/* Rocky Pass zone */}
                <path d="M160 80 L175 42 L195 34 L210 40 L220 80Z" fill="rgba(120, 85, 45, 0.55)" />
                <path d="M200 80 L208 50 L228 42 L245 48 L252 80Z" fill="rgba(110, 78, 40, 0.5)" />
                <path d="M240 80 L248 55 L260 48 L270 52 L276 80Z" fill="rgba(115, 82, 42, 0.45)" />
                <circle cx="185" cy="72" r="6" fill="rgba(120, 85, 45, 0.35)" />
                <circle cx="265" cy="74" r="4" fill="rgba(120, 85, 45, 0.3)" />

                {/* Boulder Field zone */}
                <path d="M580 80 L592 38 L612 28 L630 36 L640 80Z" fill="rgba(125, 90, 48, 0.58)" />
                <path d="M630 80 L640 45 L658 36 L672 44 L680 80Z" fill="rgba(110, 78, 40, 0.52)" />
                <path d="M670 80 L676 52 L690 44 L700 50 L708 80Z" fill="rgba(118, 84, 44, 0.48)" />
                <path d="M700 80 L710 58 L725 50 L738 56 L742 80Z" fill="rgba(105, 75, 38, 0.44)" />
                <circle cx="615" cy="70" r="7" fill="rgba(120, 85, 45, 0.32)" />
                <circle cx="730" cy="73" r="5" fill="rgba(120, 85, 45, 0.28)" />

                {/* Rough Terrain zone */}
                <path d="M1000 80 L1012 44 L1030 36 L1045 42 L1055 80Z" fill="rgba(122, 88, 46, 0.54)" />
                <path d="M1040 80 L1050 50 L1065 42 L1078 48 L1085 80Z" fill="rgba(112, 80, 42, 0.5)" />
                <path d="M1080 80 L1088 54 L1100 46 L1112 52 L1118 80Z" fill="rgba(118, 84, 44, 0.45)" />
                <circle cx="1025" cy="72" r="5" fill="rgba(120, 85, 45, 0.3)" />
                <circle cx="1105" cy="74" r="4" fill="rgba(120, 85, 45, 0.26)" />

                {/* Scattered stones */}
                <circle cx="400" cy="75" r="3" fill="rgba(130, 100, 60, 0.2)" />
                <circle cx="860" cy="76" r="3.5" fill="rgba(130, 100, 60, 0.18)" />
                <circle cx="1250" cy="74" r="3" fill="rgba(130, 100, 60, 0.16)" />
            </svg>

            {/* ── Cacti ── */}
            <svg
                className="scroll-scene__layer scroll-scene__cacti"
                viewBox="0 0 1440 100"
                preserveAspectRatio="none"
                style={{ transform: `translateX(${cactiShift}px)` }}
            >
                <g fill="rgba(55, 90, 35, 0.55)">
                    <rect x="300" y="35" width="6" height="45" rx="3" />
                    <rect x="295" y="50" width="5" height="20" rx="2.5" transform="rotate(-15, 297, 50)" />
                    <rect x="306" y="42" width="5" height="22" rx="2.5" transform="rotate(12, 308, 42)" />
                </g>
                <g fill="rgba(60, 95, 40, 0.48)">
                    <rect x="820" y="50" width="5" height="30" rx="2.5" />
                    <rect x="815" y="58" width="4" height="15" rx="2" transform="rotate(-20, 817, 58)" />
                </g>
                <g fill="rgba(50, 85, 32, 0.45)">
                    <rect x="1250" y="40" width="6" height="40" rx="3" />
                    <rect x="1256" y="48" width="5" height="18" rx="2.5" transform="rotate(18, 1258, 48)" />
                </g>
                <ellipse cx="150" cy="82" rx="14" ry="8" fill="rgba(70, 100, 45, 0.25)" />
                <ellipse cx="550" cy="85" rx="10" ry="6" fill="rgba(70, 100, 45, 0.22)" />
                <ellipse cx="1050" cy="83" rx="12" ry="7" fill="rgba(70, 100, 45, 0.24)" />
            </svg>

            {/* ── Near dunes ── */}
            <svg
                className="scroll-scene__layer scroll-scene__dunes-near"
                viewBox="0 0 1440 80"
                preserveAspectRatio="none"
                style={{ transform: `translateX(${dunesNear}px)` }}
            >
                <path d="M0 80 Q180 45 360 65 Q540 35 720 55 Q900 30 1080 50 Q1260 38 1440 48 L1600 40 L1800 55 L2000 80Z"
                    fill="rgba(170, 140, 90, 0.22)" />
            </svg>

            {/* ── Green Safari Jeep ── */}
            <div
                className="scroll-scene__vehicle"
                style={{
                    left: `${vehicleX}%`,
                    transform: `translateY(${bounce}px) rotate(${tilt}deg)`,
                }}
            >
                <svg width="160" height="86" viewBox="0 0 96 52" fill="none">
                    {/* Glow filter */}
                    <defs>
                        <filter id="jeepGlow" x="-30%" y="-30%" width="160%" height="160%">
                            <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
                            <feColorMatrix in="blur" type="matrix" values="0 0 0 0 0.3  0 0 0 0 0.69  0 0 0 0 0.31  0 0 0 0.6 0" result="glow" />
                            <feMerge>
                                <feMergeNode in="glow" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                    </defs>

                    {/* Ground shadow */}
                    <ellipse cx="48" cy="48" rx="42" ry="5" fill="rgba(40, 25, 10, 0.35)" />

                    {/* Dark outline for contrast */}
                    <path d="M14 32 L22 16 L34 11 L62 11 L74 16 L82 32Z"
                        fill="none" stroke="rgba(0,0,0,0.5)" strokeWidth="4" />

                    {/* Main body — bright green with glow */}
                    <g filter="url(#jeepGlow)">
                        <path d="M14 32 L22 16 L34 11 L62 11 L74 16 L82 32Z"
                            fill="#4CAF50" stroke="#81C784" strokeWidth="2" />

                        {/* Roof rack */}
                        <line x1="30" y1="11" x2="66" y2="11" stroke="#388E3C" strokeWidth="2" />
                        <line x1="36" y1="8" x2="60" y2="8" stroke="#388E3C" strokeWidth="1" opacity="0.7" />
                        <line x1="40" y1="8" x2="40" y2="11" stroke="#388E3C" strokeWidth="0.8" opacity="0.6" />
                        <line x1="50" y1="8" x2="50" y2="11" stroke="#388E3C" strokeWidth="0.8" opacity="0.6" />

                        {/* Windshield */}
                        <path d="M24 16 L34 11 L62 11 L72 16Z"
                            fill="rgba(200, 235, 255, 0.45)" stroke="#66BB6A" strokeWidth="0.8" />

                        {/* Side windows */}
                        <path d="M26 17 L30 12 L44 12 L44 17Z" fill="rgba(200, 235, 255, 0.35)" />
                        <path d="M52 12 L66 12 L70 17 L52 17Z" fill="rgba(200, 235, 255, 0.35)" />

                        {/* Body panel line */}
                        <line x1="16" y1="26" x2="80" y2="26" stroke="#66BB6A" strokeWidth="0.6" opacity="0.5" />

                        {/* Undercarriage */}
                        <rect x="12" y="32" width="72" height="5" rx="2" fill="#2E7D32" />
                        {/* Skid plate */}
                        <rect x="24" y="37" width="48" height="2" rx="1" fill="#1B5E20" opacity="0.8" />

                        {/* Front wheel */}
                        <circle cx="26" cy="40" r="8" fill="#212121" stroke="#66BB6A" strokeWidth="1.5" />
                        <circle cx="26" cy="40" r="4.5" fill="#333333" />
                        <circle cx="26" cy="40" r="1.5" fill="#555555" />

                        {/* Rear wheel */}
                        <circle cx="70" cy="40" r="8" fill="#212121" stroke="#66BB6A" strokeWidth="1.5" />
                        <circle cx="70" cy="40" r="4.5" fill="#333333" />
                        <circle cx="70" cy="40" r="1.5" fill="#555555" />

                        {/* Headlights — warm yellow */}
                        <rect x="80" y="22" width="4" height="6" rx="1.5" fill="#E8C84B" opacity="0.9" />
                        <rect x="80" y="22" width="4" height="6" rx="1.5" fill="white" opacity="0.25" />

                        {/* Tail light */}
                        <rect x="12" y="24" width="3" height="4" rx="1" fill="#C44040" opacity="0.7" />

                        {/* Headlight glow */}
                    </g>
                    <ellipse cx="90" cy="25" rx="10" ry="12" fill="#E8C84B" opacity="0.12" />

                    {/* Dust trail — warm sand, varies with speed */}
                    <circle cx="6" cy="38" r="5" fill={`rgba(180, 150, 100, ${dustOpacity})`} />
                    <circle cx="-6" cy="36" r="8" fill={`rgba(180, 150, 100, ${dustOpacity * 0.7})`} />
                    <circle cx="-20" cy="34" r="12" fill={`rgba(180, 150, 100, ${dustOpacity * 0.4})`} />
                    <circle cx="-38" cy="32" r="16" fill={`rgba(180, 150, 100, ${dustOpacity * 0.2})`} />
                </svg>

                {/* Speed zone indicator */}
                {isRough && (
                    <span className="scroll-scene__zone-label">
                        {zone.label}
                    </span>
                )}
            </div>

            {/* ── Ground line ── */}
            <div className="scroll-scene__ground" />
        </div>
    );
}
