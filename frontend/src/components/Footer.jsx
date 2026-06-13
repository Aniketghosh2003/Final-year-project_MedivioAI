import { Heart, Github, Mail } from 'lucide-react';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-slate-950 border-t border-slate-900 mt-auto text-slate-455">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About Section */}
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4">MedivioAI</h3>
            <p className="text-slate-400 text-sm leading-relaxed max-w-sm">
              A research prototype for AI-assisted medical image analysis,
              exploring how deep learning can surface helpful signals for
              clinicians and patients.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4">Resources</h3>
            <ul className="space-y-2.5 text-sm">
              <li>
                <a href="#" className="text-slate-400 hover:text-blue-400 transition-colors">
                  Documentation
                </a>
              </li>
              <li>
                <a href="#" className="text-slate-400 hover:text-blue-400 transition-colors">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-slate-400 hover:text-blue-400 transition-colors">
                  Terms of Service
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4">Connect</h3>
            <div className="flex space-x-4.5">
              <a
                href="#"
                className="bg-slate-900 hover:bg-slate-850 p-2.5 rounded-lg border border-slate-800 text-slate-400 hover:text-blue-450 transition-all"
                aria-label="GitHub"
              >
                <Github className="h-5 w-5" />
              </a>
              <a
                href="#"
                className="bg-slate-900 hover:bg-slate-850 p-2.5 rounded-lg border border-slate-800 text-slate-400 hover:text-blue-455 transition-all"
                aria-label="Email"
              >
                <Mail className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>

        <div className="mt-10 pt-8 border-t border-slate-900">
          <p className="text-center text-sm text-slate-400 flex items-center justify-center gap-1.5">
            Made with <Heart className="h-4 w-4 text-red-500 fill-red-500" /> for better healthcare
          </p>
          <p className="text-center text-xs text-slate-500 mt-2.5">
            © {currentYear} MedivioAI. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
