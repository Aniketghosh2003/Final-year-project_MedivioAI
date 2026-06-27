import { useState, useEffect } from 'react';
import './App.css';
import HomePage from './components/HomePage';
import SinglePrediction from './components/SinglePrediction';
import Header from './components/Header';
import Footer from './components/Footer';
import LoginRegister from './components/LoginRegister';
import Dashboard from './components/Dashboard';
import { Toaster } from 'react-hot-toast';
import Loader from './components/Loader';

const loadStoredUser = () => {
  const savedUser = localStorage.getItem('user');
  if (!savedUser) {
    return null;
  }

  try {
    return JSON.parse(savedUser);
  } catch {
    localStorage.removeItem('user');
    return null;
  }
};

export default function App() {
  const [currentPage, setCurrentPage] = useState(() => sessionStorage.getItem('currentPage') || 'home');
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [user, setUser] = useState(loadStoredUser());
  const [showLoader, setShowLoader] = useState(true);

  useEffect(() => {
    sessionStorage.setItem('currentPage', currentPage);
  }, [currentPage]);

  useEffect(() => {
    const loaderTimer = window.setTimeout(() => {
      setShowLoader(false);
    }, 10000);

    return () => window.clearTimeout(loaderTimer);
  }, []);

  const handleAuthSuccess = (newToken, newUser) => {
    setToken(newToken);
    setUser(newUser);
    localStorage.setItem('token', newToken);
    localStorage.setItem('user', JSON.stringify(newUser));
    sessionStorage.setItem('currentPage', 'dashboard');
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    sessionStorage.setItem('currentPage', 'home');
    setCurrentPage('home');
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage onNavigate={setCurrentPage} />;
      case 'auth':
        return (
          <LoginRegister 
            onAuthSuccess={handleAuthSuccess} 
            onBackToHome={() => setCurrentPage('home')} 
          />
        );
      case 'dashboard':
        return token && user ? (
          <Dashboard 
            token={token} 
            user={user} 
            onNavigate={setCurrentPage} 
          />
        ) : (
          <LoginRegister 
            onAuthSuccess={handleAuthSuccess} 
            onBackToHome={() => setCurrentPage('home')} 
          />
        );
      case 'single':
        // Scan chooser: let user pick which disease model to scan for
        return (
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16 animate-fade-in text-slate-100">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-blue-900/40 bg-blue-950/40 text-blue-300 text-xs font-semibold uppercase tracking-[0.24em] mb-5">
                Scan
              </div>
              <h1 className="text-4xl font-bold text-white mb-3">Choose a scan type</h1>
              <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
                Choose what you want to screen for, then upload a medical
                image on the next step.
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-6 lg:gap-8">
              <button
                onClick={() => setCurrentPage('scan-pneumonia')}
                className="group bg-slate-900/90 border border-slate-850 rounded-3xl p-7 text-left shadow-xl shadow-slate-950/30 hover:shadow-blue-950/20 hover:border-blue-900/40 transition-all flex flex-col justify-between cursor-pointer backdrop-blur-sm"
              >
                <div>
                  <h2 className="text-xl font-semibold text-white mb-2">Pneumonia Scan</h2>
                  <p className="text-sm text-slate-400 mb-3 leading-relaxed">
                    Analyze chest X-ray style images with a model tuned for
                    signs of pneumonia.
                  </p>
                </div>
                <span className="mt-4 inline-flex items-center text-sm font-semibold text-blue-400 group-hover:gap-1 transition-all">
                  Continue
                </span>
              </button>
              <button
                onClick={() => setCurrentPage('scan-tb')}
                className="group bg-slate-900/90 border border-slate-850 rounded-3xl p-7 text-left shadow-xl shadow-slate-950/30 hover:shadow-blue-950/20 hover:border-cyan-900/40 transition-all flex flex-col justify-between cursor-pointer backdrop-blur-sm"
              >
                <div>
                  <h2 className="text-xl font-semibold text-white mb-2">Tuberculosis Scan</h2>
                  <p className="text-sm text-slate-400 mb-3 leading-relaxed">
                    Route images through a tuberculosis-focused model to flag
                    potential TB patterns.
                  </p>
                </div>
                <span className="mt-4 inline-flex items-center text-sm font-semibold text-cyan-400 group-hover:gap-1 transition-all">
                  Continue
                </span>
              </button>
            </div>
          </div>
        );
      case 'scan-pneumonia':
        return (
          <SinglePrediction 
            mode="pneumonia" 
            token={token} 
            onBack={() => setCurrentPage('single')} 
          />
        );
      case 'scan-tb':
        return (
          <SinglePrediction 
            mode="tuberculosis" 
            token={token} 
            onBack={() => setCurrentPage('single')} 
          />
        );
      default:
        return <HomePage onNavigate={setCurrentPage} />;
    }
  };

  return (
    showLoader ? (
      <Loader />
    ) : (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-blue-600 selection:text-white">
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 3500,
          style: {
            background: '#0f172a',
            color: '#e2e8f0',
            border: '1px solid #1e293b',
          },
        }}
      />
      <Header 
        currentPage={currentPage} 
        onNavigate={setCurrentPage} 
        user={user} 
        onLogout={handleLogout} 
      />
      <main className="flex-1">
        {renderPage()}
      </main>
      <Footer />
    </div>
    )
  );
}
