import './Footer.css';

export default function Footer() {
    return (
        <footer className="footer">
            <div className="container footer__inner">
                <div className="footer__left">
                    <span className="footer__brand">desertnav</span>
                    <span className="footer__sep"></span>
                    <span className="footer__meta"></span>
                </div>
                <div className="footer__right">
                    <span className="footer__names">Made by - Samarth Shukla</span>
                </div>
            </div>
        </footer>
    );
}
