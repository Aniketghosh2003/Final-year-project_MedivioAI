import { useState } from 'react';
import { Mail, Lock, User, Calendar, Users, LogIn, UserPlus, Loader2 } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export default function LoginRegister({ onAuthSuccess, onBackToHome }) {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  // Form states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (isLogin) {
        // Login request
        const response = await axios.post(`${API_URL}/api/auth/login`, {
          email,
          password
        });
        
        if (response.data.success) {
          toast.success('Signed in successfully');
          onAuthSuccess(response.data.token, response.data.user);
        } else {
          toast.error(response.data.error || 'Login failed');
        }
      } else {
        // Registration request
        const response = await axios.post(`${API_URL}/api/auth/register`, {
          email,
          password,
          name,
          age: age ? parseInt(age, 10) : null,
          gender
        });

        if (response.data.success) {
          toast.success('Account created successfully');
          onAuthSuccess(response.data.token, response.data.user);
        } else {
          toast.error(response.data.error || 'Registration failed');
        }
      }
    } catch (err) {
      toast.error(err.response?.data?.error || 'Failed to connect to the authentication server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto my-12 px-4 sm:px-6 animate-fade-in text-slate-100">
      <div className="bg-slate-900 rounded-3xl shadow-xl border border-slate-850 overflow-hidden">
        {/* Tab Header */}
        <div className="flex border-b border-slate-850 bg-slate-950/40">
          <button
            type="button"
            onClick={() => {
              setIsLogin(true);
            }}
            className={`flex-1 py-4 text-center font-semibold text-sm transition-all flex items-center justify-center gap-2 border-b-2 cursor-pointer ${
              isLogin
                ? 'border-blue-500 text-blue-400 bg-slate-900'
                : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-slate-950/10'
            }`}
          >
            <LogIn className="h-4 w-4" />
            Sign In
          </button>
          <button
            type="button"
            onClick={() => {
              setIsLogin(false);
            }}
            className={`flex-1 py-4 text-center font-semibold text-sm transition-all flex items-center justify-center gap-2 border-b-2 cursor-pointer ${
              !isLogin
                ? 'border-blue-500 text-blue-400 bg-slate-900'
                : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-slate-950/10'
            }`}
          >
            <UserPlus className="h-4 w-4" />
            Create Account
          </button>
        </div>

        {/* Form Body */}
        <div className="p-8">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-white">
              {isLogin ? 'Welcome Back' : 'Get Started'}
            </h2>
            <p className="text-xs text-slate-450 mt-1">
              {isLogin
                ? 'Access your medical records and scan history'
                : 'Create an account to securely save scan reports'}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4.5">
            {/* Name - Register Only */}
            {!isLogin && (
              <div>
                <label className="block text-xs font-semibold text-slate-450 uppercase tracking-wider mb-1.5">
                  Full Name
                </label>
                <div className="relative">
                  <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-500">
                    <User className="h-5 w-5" />
                  </span>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="John Doe"
                    className="block w-full pl-10 pr-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl focus:bg-slate-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-900/10 transition-all text-sm text-white placeholder-slate-650 outline-hidden"
                  />
                </div>
              </div>
            )}

            {/* Email - Both */}
            <div>
              <label className="block text-xs font-semibold text-slate-455 uppercase tracking-wider mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-500">
                  <Mail className="h-5 w-5" />
                </span>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="john@example.com"
                  className="block w-full pl-10 pr-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl focus:bg-slate-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-900/10 transition-all text-sm text-white placeholder-slate-650 outline-hidden"
                />
              </div>
            </div>

            {/* Password - Both */}
            <div>
              <label className="block text-xs font-semibold text-slate-455 uppercase tracking-wider mb-1.5">
                Password
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-slate-500">
                  <Lock className="h-5 w-5" />
                </span>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="block w-full pl-10 pr-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl focus:bg-slate-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-900/10 transition-all text-sm text-white placeholder-slate-650 outline-hidden"
                />
              </div>
            </div>

            {/* Age & Gender - Register Only */}
            {!isLogin && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-455 uppercase tracking-wider mb-1.5">
                    Age
                  </label>
                  <div className="relative">
                    <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-slate-500">
                      <Calendar className="h-4 w-4" />
                    </span>
                    <input
                      type="number"
                      min="0"
                      max="150"
                      value={age}
                      onChange={(e) => setAge(e.target.value)}
                      placeholder="28"
                      className="block w-full pl-9 pr-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl focus:bg-slate-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-900/10 transition-all text-sm text-white placeholder-slate-650 outline-hidden"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-455 uppercase tracking-wider mb-1.5">
                    Gender
                  </label>
                  <div className="relative">
                    <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-slate-500">
                      <Users className="h-4 w-4" />
                    </span>
                    <select
                      value={gender}
                      onChange={(e) => setGender(e.target.value)}
                      className="block w-full pl-9 pr-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl focus:bg-slate-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-900/10 transition-all text-sm text-white outline-hidden appearance-none"
                    >
                      <option value="" className="bg-slate-950 text-slate-400">Select</option>
                      <option value="Male" className="bg-slate-950 text-white">Male</option>
                      <option value="Female" className="bg-slate-950 text-white">Female</option>
                      <option value="Other" className="bg-slate-950 text-white">Other</option>
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg hover:shadow-blue-500/10 flex items-center justify-center gap-2 mt-6 cursor-pointer"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  {isLogin ? 'Signing In...' : 'Registering...'}
                </>
              ) : (
                <>
                  {isLogin ? <LogIn className="h-5 w-5" /> : <UserPlus className="h-5 w-5" />}
                  {isLogin ? 'Sign In' : 'Create Account'}
                </>
              )}
            </button>
          </form>

          {/* Back link */}
          <div className="text-center mt-6">
            <button
              type="button"
              onClick={onBackToHome}
              className="text-xs font-semibold text-slate-500 hover:text-white transition-all cursor-pointer"
            >
              ← Back to Homepage
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
