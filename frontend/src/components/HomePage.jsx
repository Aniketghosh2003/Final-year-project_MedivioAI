import { Brain, Zap, Shield, ArrowRight, Activity, FileSearch, Sparkles } from 'lucide-react';

export default function HomePage({ onNavigate }) {
  const features = [
    {
      icon: Brain,
      title: 'Intelligent Imaging Insight',
      description:
        'Deep learning models that highlight subtle patterns in medical images to support earlier, more confident decisions.',
      color: 'bg-blue-100 text-blue-600',
    },
    {
      icon: Zap,
      title: 'Near‑Instant Feedback',
      description:
        'Upload, analyze and review findings in just a few seconds, right at the point of care.',
      color: 'bg-emerald-100 text-emerald-600',
    },
    {
      icon: Shield,
      title: 'Built for Clinicians',
      description:
        'Designed as a companion to clinical judgment, helping teams triage, prioritize and communicate findings.',
      color: 'bg-cyan-100 text-cyan-600',
    },
    {
      icon: FileSearch,
      title: 'Future‑Ready Platform',
      description:
        'A flexible foundation ready to host additional disease models and imaging modalities as the project grows.',
      color: 'bg-indigo-100 text-indigo-600',
    },
  ];

  return (
    <div className="animate-fade-in">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-28">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Column */}
            <div className="space-y-8">
              <div className="inline-flex items-center space-x-2 bg-blue-50 border border-blue-200 rounded-full px-4 py-2">
                <Sparkles className="h-4 w-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-700">
                  AI-Powered Medical Image Analysis:
                </span>
              </div>

              <h1 className="text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
                Medical Image Analysis
                <span className="block text-blue-600 mt-2">AI Support for Healthcare Imaging</span>
              </h1>

              <p className="text-xl text-gray-600 leading-relaxed">
                Harness the power of artificial intelligence to analyze medical
                images with clinical-grade insight. This prototype currently
                focuses on pneumonia detection as a first use case, and is
                designed to grow into a broader medical imaging assistant.
              </p>

              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={() => onNavigate("single")}
                  className="group bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
                >
                  Start Scan
                  <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>

              {/* Highlights */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-8 text-sm text-gray-600">
                <div className="rounded-2xl bg-white/60 border border-gray-200 p-4 backdrop-blur">
                  <p className="font-semibold text-gray-900 mb-1">Human‑in‑the‑Loop</p>
                  <p>Designed to complement, never replace, expert clinical judgment.</p>
                </div>
                <div className="rounded-2xl bg-white/60 border border-gray-200 p-4 backdrop-blur">
                  <p className="font-semibold text-gray-900 mb-1">Transparent Signals</p>
                  <p>Clear probability scores to understand how the model sees each case.</p>
                </div>
                <div className="rounded-2xl bg-white/60 border border-gray-200 p-4 backdrop-blur">
                  <p className="font-semibold text-gray-900 mb-1">Research Prototype</p>
                  <p>Built for learning, experimentation and future expansion to more diseases.</p>
                </div>
              </div>
            </div>

            {/* Right Column - Visual */}
            <div className="relative">
              <div className="relative bg-linear-to-br from-blue-500 to-cyan-500 rounded-3xl p-8 shadow-2xl">
                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center space-x-3">
                      <div className="bg-white/20 p-2 rounded-lg">
                        <Activity className="h-6 w-6 text-white" />
                      </div>
                      <div className="text-white">
                        <div className="text-sm opacity-80">
                          Analysis Status
                        </div>
                        <div className="font-semibold">Ready to Scan</div>
                      </div>
                    </div>
                    <div className="bg-emerald-500 text-white text-xs font-semibold px-3 py-1 rounded-full">
                      Online
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="bg-white/10 rounded-xl p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white/80 text-sm">
                          Current Focus
                        </span>
                        <span className="text-white font-semibold">Pneumonia detection</span>
                      </div>
                      <div className="bg-white/20 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-white/80 h-full rounded-full"
                          style={{ width: "60%" }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-white/10 rounded-xl p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white/80 text-sm">
                          Prototype Maturity
                        </span>
                        <span className="text-white font-semibold">
                          Research stage
                        </span>
                      </div>
                      <div className="bg-white/20 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-emerald-400 h-full rounded-full"
                          style={{ width: "70%" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Decorative Elements */}
              <div className="absolute -top-4 -right-4 w-24 h-24 bg-yellow-400 rounded-full opacity-20 blur-2xl"></div>
              <div className="absolute -bottom-4 -left-4 w-32 h-32 bg-blue-600 rounded-full opacity-20 blur-2xl"></div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Why Choose MedivioAI?
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Cutting-edge technology meets healthcare excellence
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="group bg-gray-50 rounded-2xl p-6 hover:bg-white hover:shadow-xl transition-all duration-300 border border-gray-100"
              >
                <div
                  className={`${feature.color} w-14 h-14 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}
                >
                  <feature.icon className="h-7 w-7" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
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
          <div className="bg-linear-to-r from-blue-600 to-cyan-600 rounded-3xl p-12 shadow-2xl text-center">
            <h2 className="text-4xl font-bold text-white mb-4">
              Ready to Analyze Medical Images?
            </h2>
            <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
              Upload medical images and receive instant AI-powered analysis
              with detailed confidence scores. Pneumonia detection is the
              first supported scenario, with more conditions planned for
              future releases.
            </p>
            <button
              onClick={() => onNavigate("single")}
              className="group bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold hover:bg-blue-50 transition-all shadow-lg hover:shadow-xl inline-flex items-center gap-2"
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
