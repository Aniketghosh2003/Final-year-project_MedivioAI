import { Brain, Zap, Shield, ArrowRight, Activity, FileSearch, Sparkles } from 'lucide-react';

export default function HomePage({ onNavigate }) {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced deep learning models trained on thousands of X-ray images for accurate pneumonia detection.',
      color: 'bg-blue-100 text-blue-600',
    },
    {
      icon: Zap,
      title: 'Instant Results',
      description: 'Get predictions in seconds with confidence scores and detailed probability breakdowns.',
      color: 'bg-emerald-100 text-emerald-600',
    },
    {
      icon: Shield,
      title: 'Clinical Grade',
      description: 'Trained on validated medical datasets to assist healthcare professionals in diagnosis.',
      color: 'bg-cyan-100 text-cyan-600',
    },
    {
      icon: FileSearch,
      title: 'Batch Processing',
      description: 'Analyze multiple X-ray images simultaneously for efficient workflow management.',
      color: 'bg-indigo-100 text-indigo-600',
    },
  ];

  const stats = [
    { label: 'Accuracy', value: '95%+' },
    { label: 'Processing Time', value: '<3s' },
    { label: 'Images Analyzed', value: '10K+' },
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
                  AI-Powered Medical Diagnostics
                </span>
              </div>

              <h1 className="text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
                Pneumonia Detection
                <span className="block text-blue-600 mt-2">Made Simple</span>
              </h1>

              <p className="text-xl text-gray-600 leading-relaxed">
                Harness the power of artificial intelligence to detect pneumonia 
                from chest X-rays with clinical-grade accuracy. Fast, reliable, 
                and designed for healthcare professionals.
              </p>

              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={() => onNavigate('single')}
                  className="group bg-blue-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
                >
                  Start Analysis
                  <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </button>
                <button
                  onClick={() => onNavigate('batch')}
                  className="bg-white text-gray-700 px-8 py-4 rounded-xl font-semibold hover:bg-gray-50 transition-all border-2 border-gray-200 hover:border-gray-300"
                >
                  Batch Processing
                </button>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-6 pt-8">
                {stats.map((stat, index) => (
                  <div key={index} className="text-center">
                    <div className="text-3xl font-bold text-gray-900">{stat.value}</div>
                    <div className="text-sm text-gray-600 mt-1">{stat.label}</div>
                  </div>
                ))}
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
                        <div className="text-sm opacity-80">Analysis Status</div>
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
                        <span className="text-white/80 text-sm">Model Accuracy</span>
                        <span className="text-white font-semibold">95.2%</span>
                      </div>
                      <div className="bg-white/20 rounded-full h-2 overflow-hidden">
                        <div className="bg-white h-full rounded-full" style={{ width: '95%' }}></div>
                      </div>
                    </div>

                    <div className="bg-white/10 rounded-xl p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white/80 text-sm">Processing Speed</span>
                        <span className="text-white font-semibold">2.8s avg</span>
                      </div>
                      <div className="bg-white/20 rounded-full h-2 overflow-hidden">
                        <div className="bg-emerald-400 h-full rounded-full" style={{ width: '88%' }}></div>
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
                <div className={`${feature.color} w-14 h-14 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
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
              Ready to Get Started?
            </h2>
            <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
              Upload chest X-ray images and receive instant AI-powered analysis 
              with detailed confidence scores.
            </p>
            <button
              onClick={() => onNavigate('single')}
              className="group bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold hover:bg-blue-50 transition-all shadow-lg hover:shadow-xl inline-flex items-center gap-2"
            >
              Start Your First Scan
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
