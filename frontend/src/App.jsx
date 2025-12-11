import { useState } from 'react';
import './App.css';
import HomePage from './components/HomePage';
import SinglePrediction from './components/SinglePrediction';
import Header from './components/Header';
import Footer from './components/Footer';

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage onNavigate={setCurrentPage} />;
      case 'single':
        // Scan chooser: let user pick which disease model to scan for
        return (
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-16 animate-fade-in">
            <div className="text-center mb-10">
              <h1 className="text-4xl font-bold text-gray-900 mb-3">Scan</h1>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Choose what you want to screen for, then upload a medical
                image on the next step.
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <button
                onClick={() => setCurrentPage('scan-pneumonia')}
                className="group bg-white border border-blue-100 rounded-2xl p-6 text-left shadow-sm hover:shadow-lg hover:border-blue-300 transition-all flex flex-col justify-between"
              >
                <div>
                  <h2 className="text-xl font-semibold text-gray-900 mb-2">Pneumonia Scan</h2>
                  <p className="text-sm text-gray-600 mb-3">
                    Analyze chest X-ray style images with a model tuned for
                    signs of pneumonia.
                  </p>
                </div>
                <span className="mt-4 inline-flex items-center text-sm font-medium text-blue-600 group-hover:gap-1">
                  Continue
                </span>
              </button>
              <button
                onClick={() => setCurrentPage('scan-tb')}
                className="group bg-white border border-blue-100 rounded-2xl p-6 text-left shadow-sm hover:shadow-lg hover:border-blue-300 transition-all flex flex-col justify-between"
              >
                <div>
                  <h2 className="text-xl font-semibold text-gray-900 mb-2">Tuberculosis Scan</h2>
                  <p className="text-sm text-gray-600 mb-3">
                    Route images through a tuberculosis-focused model to flag
                    potential TB patterns.
                  </p>
                </div>
                <span className="mt-4 inline-flex items-center text-sm font-medium text-blue-600 group-hover:gap-1">
                  Continue
                </span>
              </button>
            </div>
          </div>
        );
      case 'scan-pneumonia':
        return <SinglePrediction mode="pneumonia" onBack={() => setCurrentPage('single')} />;
      case 'scan-tb':
        return <SinglePrediction mode="tuberculosis" onBack={() => setCurrentPage('single')} />;
      default:
        return <HomePage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-linear-to-br from-slate-50 to-blue-50">
      <Header currentPage={currentPage} onNavigate={setCurrentPage} />
      <main className="flex-1">
        {renderPage()}
      </main>
      <Footer />
    </div>
  );
}
