import Navbar from './components/Navbar';
import Hero from './components/Hero';
import HowItWorks from './components/HowItWorks';
import Upload from './components/Upload';
import Metrics from './components/Metrics';
import Legend from './components/Legend';
import Footer from './components/Footer';
import ShaderBg from './components/ShaderBg';
import ScrollScene from './components/ScrollScene';

function App() {
  return (
    <>
      <ShaderBg />
      <ScrollScene />
      <Navbar />
      <Hero />
      <HowItWorks />
      <Upload />
      <Metrics />
      <Legend />
      <Footer />
    </>
  );
}

export default App;
