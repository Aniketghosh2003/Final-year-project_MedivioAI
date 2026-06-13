import { Brain, Zap, Shield, ArrowRight, Activity, FileSearch, Sparkles } from 'lucide-react';

export default function HomePage({ onNavigate }) {
  const features = [
    {
      icon: Brain,
      title: 'Intelligent Imaging Insight',
      description:
        'Deep learning models that highlight subtle patterns in medical images to support earlier, more confident decisions.',
      color: 'bg-blue-950/50 text-blue-400 border border-blue-900/30',
    },
    {
      icon: Zap,
      title: 'Near‑Instant Feedback',
      description:
        'Upload, analyze and review findings in just a few seconds, right at the point of care.',
      color: 'bg-emerald-950/50 text-emerald-400 border border-emerald-900/30',
    },
    {
      icon: Shield,
      title: 'Built for Clinicians',
      description:
        'Designed as a companion to clinical judgment, helping teams triage, prioritize and communicate findings.',
      color: 'bg-cyan-950/50 text-cyan-400 border border-cyan-900/30',
    },
    {
      icon: FileSearch,
      title: 'Future‑Ready Platform',
      description:
        'A flexible foundation ready to host additional disease models and imaging modalities as the project grows.',
      color: 'bg-indigo-950/50 text-indigo-400 border border-indigo-900/30',
    },
  ];

  return (
    <div className="animate-fade-in text-slate-100">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-28">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Column */}
            <div className="space-y-8">
              <div className="inline-flex items-center space-x-2 bg-blue-950/60 border border-blue-900/40 rounded-full px-4 py-2">
                <Sparkles className="h-4 w-4 text-blue-400 animate-pulse" />
                <span className="text-sm font-medium text-blue-300">
                  AI-Powered Medical Image Analysis
                </span>
              </div>

              <h1 className="text-5xl lg:text-6xl font-bold text-white leading-tight">
                Medical Image Analysis
                <span className="block text-blue-400 mt-2 bg-clip-text bg-linear-to-r from-blue-400 to-cyan-300">
                  AI Support for Clinicians
                </span>
              </h1>

              <p className="text-lg text-slate-400 leading-relaxed">
                Harness the power of artificial intelligence to analyze medical
                images with clinical-grade insight. This prototype currently
                focuses on pneumonia detection as a first usecase, and is
                designed to grow into a broader medical imaging assistant.
              </p>

              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={() => onNavigate("single")}
                  className="group bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 transition-all shadow-lg shadow-blue-500/20 hover:shadow-xl hover:shadow-blue-500/35 flex items-center justify-center gap-2 cursor-pointer"
                >
                  Start Scan
                  <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>

              {/* Highlights */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-8 text-sm text-slate-400">
                <div className="rounded-2xl bg-slate-900/40 border border-slate-850 p-4 backdrop-blur-sm">
                  <p className="font-semibold text-white mb-1">Human‑in‑the‑Loop</p>
                  <p className="text-xs">Complement, never replace, expert clinical judgment.</p>
                </div>
                <div className="rounded-2xl bg-slate-900/40 border border-slate-850 p-4 backdrop-blur-sm">
                  <p className="font-semibold text-white mb-1">Transparent Signals</p>
                  <p className="text-xs">Clear probability scores to understand model predictions.</p>
                </div>
                <div className="rounded-2xl bg-slate-900/40 border border-slate-850 p-4 backdrop-blur-sm">
                  <p className="font-semibold text-white mb-1">Research Prototype</p>
                  <p className="text-xs">Built for learning and expansion to more diseases.</p>
                </div>
              </div>
            </div>

            {/* Right Column - Visual */}
            <div className="relative">
              <div className="relative bg-linear-to-br from-blue-600 to-indigo-950 rounded-3xl p-8 shadow-2xl border border-blue-900/30">
                <div className="bg-slate-950/80 backdrop-blur-md rounded-2xl p-6 border border-slate-850 shadow-inner">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-950 p-2.5 rounded-lg border border-blue-900/30 text-blue-400">
                        <Activity className="h-6 w-6" />
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">
                          Analysis Status
                        </div>
                        <div className="font-semibold text-white">Ready to Scan</div>
                      </div>
                    </div>
                    <div className="bg-emerald-950 text-emerald-400 border border-emerald-900/30 text-xs font-semibold px-3 py-1 rounded-full">
                      Online
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-850">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-slate-400 text-sm">
                          Current Focus
                        </span>
                        <span className="text-blue-400 font-semibold text-sm">Pneumonia detection</span>
                      </div>
                      <div className="bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-900">
                        <div
                          className="bg-blue-500 h-full rounded-full"
                          style={{ width: "100%" }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-850">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-slate-400 text-sm">
                          Tuberculosis Scan
                        </span>
                        <span className="text-cyan-400 font-semibold text-sm">
                          Active
                        </span>
                      </div>
                      <div className="bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-900">
                        <div
                          className="bg-cyan-500 h-full rounded-full"
                          style={{ width: "100%" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Decorative Elements */}
              <div className="absolute -top-4 -right-4 w-24 h-24 bg-blue-500 rounded-full opacity-10 blur-2xl"></div>
              <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-indigo-500 rounded-full opacity-15 blur-3xl"></div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-slate-950 border-t border-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-4">
              Why Choose MedivioAI?
            </h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Cutting-edge technology meets healthcare excellence
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="group bg-slate-900/30 hover:bg-slate-900/60 border border-slate-900 hover:border-slate-800 rounded-2xl p-6 transition-all duration-300 shadow-sm"
              >
                <div
                  className={`${feature.color} w-14 h-14 rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform`}
                >
                  <feature.icon className="h-7 w-7" />
                </div>
                <h3 className="text-lg font-bold text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-sm text-slate-400 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-linear-to-r from-blue-900/40 to-indigo-950/40 rounded-3xl p-10 md:p-12 shadow-xl border border-blue-950/60 text-center">
            <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-4">
              Ready to Analyze Medical Images?
            </h2>
            <p className="text-sm md:text-base text-slate-300 mb-8 max-w-2xl mx-auto leading-relaxed">
              Upload medical images and receive instant AI-powered analysis
              with detailed confidence scores. Pneumonia detection is the
              first supported scenario, with more conditions planned for
              future releases.
            </p>
            <button
              onClick={() => onNavigate("single")}
              className="group bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 transition-all shadow-md shadow-blue-500/10 hover:shadow-lg hover:shadow-blue-500/20 inline-flex items-center gap-2 cursor-pointer animate-pulse-subtle"
            >
              Run Your First Analysis
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
