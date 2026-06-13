import { Activity, Menu, X, LogOut, User as UserIcon, LogIn } from 'lucide-react';
import { useState } from 'react';

export default function Header({ currentPage, onNavigate, user, onLogout }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'single', label: 'Scan' },
  ];

  if (user) {
    navItems.push({ id: 'dashboard', label: 'Dashboard' });
  }

  return (
    <header className="bg-slate-950/85 backdrop-blur-md border-b border-slate-900 sticky top-0 z-50">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div 
            className="flex items-center space-x-2 cursor-pointer shrink-0 animate-fade-in"
            onClick={() => onNavigate('home')}
          >
            <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-500/20">
              <Activity className="h-6 w-6 text-white" />
            </div>
            <span className="text-xl font-bold text-white tracking-wide">
              Medivio<span className="text-blue-500">AI</span>
            </span>
          </div>

          {/* Desktop Navigation & Auth */}
          <div className="hidden md:flex items-center space-x-8">
            <div className="flex items-center space-x-1">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all border cursor-pointer ${
                    currentPage === item.id
                      ? 'bg-slate-900 text-blue-400 border-blue-900/30 shadow-xs'
                      : 'text-slate-400 hover:bg-slate-900/50 hover:text-slate-100 border-transparent'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Auth Actions */}
            <div className="flex items-center border-l border-slate-800 pl-8">
              {user ? (
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2 text-sm text-slate-300 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800 shadow-xs animate-fade-in">
                    <UserIcon className="h-4 w-4 text-blue-500" />
                    <span className="font-semibold">{user.name}</span>
                  </div>
                  <button
                    onClick={onLogout}
                    className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-semibold border border-slate-850 text-slate-400 hover:bg-red-950/30 hover:text-red-400 hover:border-red-900/30 transition-all cursor-pointer"
                  >
                    <LogOut className="h-4 w-4" />
                    Logout
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => onNavigate('auth')}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700 transition-all shadow-md hover:shadow-blue-500/10 cursor-pointer"
                >
                  <LogIn className="h-4 w-4" />
                  Login / Register
                </button>
              )}
            </div>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg text-slate-400 hover:bg-slate-900 cursor-pointer"
          >
            {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-slate-900 space-y-3">
            <div className="space-y-1">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    onNavigate(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`block w-full text-left px-4 py-2.5 rounded-lg font-medium text-sm transition-all border cursor-pointer ${
                    currentPage === item.id
                      ? 'bg-slate-900 text-blue-400 font-semibold border-blue-900/30'
                      : 'text-slate-400 hover:bg-slate-900/50 hover:text-slate-100 border-transparent'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Mobile Auth actions */}
            <div className="pt-3 border-t border-slate-900 px-4">
              {user ? (
                <div className="space-y-3">
                  <div className="flex items-center space-x-2 text-sm text-slate-300 font-semibold">
                    <UserIcon className="h-4 w-4 text-blue-500" />
                    <span>{user.name}</span>
                  </div>
                  <button
                    onClick={() => {
                      onLogout();
                      setMobileMenuOpen(false);
                    }}
                    className="flex w-full items-center justify-center gap-1.5 px-4 py-2.5 rounded-lg text-sm font-semibold bg-red-950/20 text-red-400 border border-red-900/30 hover:bg-red-900/30 transition-all cursor-pointer"
                  >
                    <LogOut className="h-4 w-4" />
                    Logout
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => {
                    onNavigate('auth');
                    setMobileMenuOpen(false);
                  }}
                  className="flex w-full items-center justify-center gap-1.5 px-4 py-2.5 rounded-lg text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700 transition-all cursor-pointer"
                >
                  <LogIn className="h-4 w-4" />
                  Login / Register
                </button>
              )}
            </div>
          </div>
        )}
      </nav>
    </header>
  );
}
