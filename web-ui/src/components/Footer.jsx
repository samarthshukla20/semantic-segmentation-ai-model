import './Footer.css';

export default function Footer() {
    return (
        <footer className="footer">
            <div className="container footer__inner">
                <div className="footer__left">
                    <span className="footer__brand">desertnav</span>
                    <span className="footer__sep">·</span>
                    <span className="footer__meta">Startathon 2025</span>
                </div>
                <div className="footer__right">
                    <span className="footer__names">Dhruv Bajpai, Samarth Shukla, Kshitij Trivedi</span>
                </div>
            </div>
        </footer>
    );
}
