import { useState, useEffect } from 'react';
import './App.css';
import HomePage from './components/HomePage';
import SinglePrediction from './components/SinglePrediction';
import Header from './components/Header';
import Footer from './components/Footer';
import LoginRegister from './components/LoginRegister';
import Dashboard from './components/Dashboard';

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [user, setUser] = useState(null);

  // Parse user info on startup
  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (e) {
        console.error('Error parsing stored user:', e);
        localStorage.removeItem('user');
      }
    }
  }, []);

  const handleAuthSuccess = (newToken, newUser) => {
    setToken(newToken);
    setUser(newUser);
    localStorage.setItem('token', newToken);
    localStorage.setItem('user', JSON.stringify(newUser));
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
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
                className="group bg-white border border-blue-100 rounded-2xl p-6 text-left shadow-sm hover:shadow-lg hover:border-blue-300 transition-all flex flex-col justify-between cursor-pointer"
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
                className="group bg-white border border-blue-100 rounded-2xl p-6 text-left shadow-sm hover:shadow-lg hover:border-blue-300 transition-all flex flex-col justify-between cursor-pointer"
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
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-blue-600 selection:text-white">
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
  );
}
