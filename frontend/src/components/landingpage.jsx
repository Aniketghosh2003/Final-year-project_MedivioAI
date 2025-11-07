import { useState } from "react";
import {
  Upload,
  Activity,
  AlertCircle,
  CheckCircle,
  Loader2,
  XCircle,
  FileImage,
  Brain,
  TrendingUp,
  Shield,
  Zap,
  ChevronRight,
  BarChart3,
} from "lucide-react";

// Header Component
const Header = () => (
  <header className="relative bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-800 text-white overflow-hidden">
    <div className="absolute inset-0 bg-grid-white/[0.05] bg-[size:20px_20px]" />
    <div className="absolute inset-0 bg-gradient-to-b from-transparent to-blue-900/20" />

    <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="flex items-center gap-4 mb-6">
        <div className="p-3 bg-white/10 backdrop-blur-sm rounded-2xl">
          <Activity className="w-10 h-10" />
        </div>
        <div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
            AI Pneumonia Detector
          </h1>
          <p className="text-blue-100 mt-2 text-lg">
            Advanced Medical Image Analysis powered by Deep Learning
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <Brain className="w-5 h-5 text-blue-200" />
            <span className="text-sm text-blue-200">Accuracy</span>
          </div>
          <p className="text-3xl font-bold">89%</p>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-green-200" />
            <span className="text-sm text-blue-200">Sensitivity</span>
          </div>
          <p className="text-3xl font-bold">94%</p>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-5 h-5 text-purple-200" />
            <span className="text-sm text-blue-200">Specificity</span>
          </div>
          <p className="text-3xl font-bold">82%</p>
        </div>
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-5 h-5 text-yellow-200" />
            <span className="text-sm text-blue-200">F1-Score</span>
          </div>
          <p className="text-3xl font-bold">89%</p>
        </div>
      </div>
    </div>
  </header>
);

// Upload Component
const UploadSection = ({
  preview,
  selectedFile,
  onFileSelect,
  onAnalyze,
  onReset,
  loading,
}) => {
  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      const fakeEvent = { target: { files: [file] } };
      onFileSelect(fakeEvent);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-100 rounded-lg">
          <FileImage className="w-6 h-6 text-blue-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900">Upload X-Ray Image</h2>
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className="relative border-2 border-dashed border-gray-300 rounded-2xl p-12 text-center hover:border-blue-500 hover:bg-blue-50/50 transition-all duration-300 group"
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept="image/*"
          onChange={onFileSelect}
        />
        <label htmlFor="file-upload" className="cursor-pointer">
          <div className="flex flex-col items-center">
            <div className="p-4 bg-blue-100 rounded-full group-hover:bg-blue-200 transition-colors mb-4">
              <Upload className="w-10 h-10 text-blue-600" />
            </div>
            <p className="text-lg font-medium text-gray-700 mb-2">
              Drop your X-ray image here
            </p>
            <p className="text-sm text-gray-500 mb-4">
              or click to browse from your device
            </p>
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
              <Upload className="w-4 h-4" />
              Select File
            </div>
          </div>
        </label>
      </div>

      {preview && (
        <div className="mt-6 animate-in fade-in duration-500">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-700">Preview</h3>
            <span className="text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
              {selectedFile?.name}
            </span>
          </div>
          <div className="relative rounded-xl overflow-hidden bg-gray-900 border-2 border-gray-200">
            <img src={preview} alt="Preview" className="w-full h-auto" />
          </div>
        </div>
      )}

      <div className="mt-8 flex gap-3">
        <button
          onClick={onAnalyze}
          disabled={!selectedFile || loading}
          className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-xl font-semibold hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
        >
          {loading ? (
            <>
              <Loader2 className="w-6 h-6 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Brain className="w-6 h-6" />
              <span>Analyze Image</span>
              <ChevronRight className="w-5 h-5" />
            </>
          )}
        </button>

        {selectedFile && (
          <button
            onClick={onReset}
            className="px-8 py-4 rounded-xl font-semibold border-2 border-gray-300 hover:bg-gray-50 hover:border-gray-400 transition-all duration-300"
          >
            Reset
          </button>
        )}
      </div>

      <div className="mt-6 p-4 bg-blue-50 rounded-xl border border-blue-100">
        <div className="flex gap-3">
          <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-blue-900 mb-1">
              Important Guidelines
            </p>
            <ul className="text-xs text-blue-700 space-y-1">
              <li>
                • Upload clear chest X-ray images in JPG, PNG, or JPEG format
              </li>
              <li>• Maximum file size: 10MB</li>
              <li>• Best results with frontal view chest X-rays</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

// Results Component
const ResultsSection = ({ result }) => {
  const isNormal = result.prediction === "NORMAL";

  return (
    <div className="animate-in fade-in slide-in-from-bottom duration-700">
      <div
        className={`rounded-2xl shadow-xl p-8 border-2 ${
          isNormal
            ? "bg-gradient-to-br from-green-50 to-emerald-50 border-green-200"
            : "bg-gradient-to-br from-orange-50 to-red-50 border-orange-200"
        }`}
      >
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-4">
            <div
              className={`p-3 rounded-2xl ${
                isNormal ? "bg-green-100" : "bg-orange-100"
              }`}
            >
              {isNormal ? (
                <CheckCircle className="w-8 h-8 text-green-600" />
              ) : (
                <AlertCircle className="w-8 h-8 text-orange-600" />
              )}
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600 mb-1">
                Diagnosis Result
              </p>
              <h3 className="text-4xl font-bold text-gray-900">
                {result.prediction}
              </h3>
            </div>
          </div>
          <div
            className={`px-4 py-2 rounded-xl ${
              isNormal
                ? "bg-green-200 text-green-800"
                : "bg-orange-200 text-orange-800"
            }`}
          >
            <p className="text-sm font-bold">{result.confidence}% Confident</p>
          </div>
        </div>

        <div
          className={`h-1 w-full rounded-full mb-6 ${
            isNormal ? "bg-green-200" : "bg-orange-200"
          }`}
        >
          <div
            className={`h-1 rounded-full transition-all duration-1000 ${
              isNormal ? "bg-green-600" : "bg-orange-600"
            }`}
            style={{ width: `${result.confidence}%` }}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white/60 backdrop-blur rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm font-medium text-gray-700">Normal</span>
            </div>
            <p className="text-3xl font-bold text-gray-900">
              {result.probability.normal}%
            </p>
          </div>
          <div className="bg-white/60 backdrop-blur rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
              <span className="text-sm font-medium text-gray-700">
                Pneumonia
              </span>
            </div>
            <p className="text-3xl font-bold text-gray-900">
              {result.probability.pneumonia}%
            </p>
          </div>
        </div>
      </div>

      <div className="mt-6 bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="w-6 h-6 text-blue-600" />
          <h3 className="text-xl font-bold text-gray-900">Detailed Analysis</h3>
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">
                Normal Probability
              </span>
              <span className="text-sm font-bold text-gray-900">
                {result.probability.normal}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div
                className="bg-gradient-to-r from-green-400 to-green-600 h-3 rounded-full transition-all duration-1000 ease-out"
                style={{ width: `${result.probability.normal}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">
                Pneumonia Probability
              </span>
              <span className="text-sm font-bold text-gray-900">
                {result.probability.pneumonia}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div
                className="bg-gradient-to-r from-orange-400 to-red-600 h-3 rounded-full transition-all duration-1000 ease-out"
                style={{ width: `${result.probability.pneumonia}%` }}
              />
            </div>
          </div>
        </div>

        <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
          <div className="flex gap-3">
            <Shield className="w-5 h-5 text-yellow-700 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-yellow-900 mb-1">
                Medical Disclaimer
              </p>
              <p className="text-xs text-yellow-800">
                This AI analysis is for educational and research purposes only.
                Always consult with qualified healthcare professionals for
                medical diagnosis and treatment decisions.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Error Component
const ErrorMessage = ({ error, onDismiss }) => (
  <div className="animate-in fade-in slide-in-from-top duration-500">
    <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6 shadow-lg">
      <div className="flex items-start gap-4">
        <div className="p-2 bg-red-100 rounded-lg">
          <XCircle className="w-6 h-6 text-red-600" />
        </div>
        <div className="flex-1">
          <h3 className="font-bold text-red-900 mb-2 text-lg">
            Error Occurred
          </h3>
          <p className="text-red-700">{error}</p>
        </div>
        <button
          onClick={onDismiss}
          className="text-red-400 hover:text-red-600 transition-colors"
        >
          <XCircle className="w-5 h-5" />
        </button>
      </div>
    </div>
  </div>
);

// Placeholder Component
const ResultsPlaceholder = () => (
  <div className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl shadow-xl p-12 text-center border border-gray-200">
    <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
      <Brain className="w-10 h-10 text-blue-600" />
    </div>
    <h3 className="text-2xl font-bold text-gray-900 mb-3">Ready to Analyze</h3>
    <p className="text-gray-600 max-w-md mx-auto">
      Upload a chest X-ray image and click "Analyze Image" to get instant
      AI-powered diagnosis results
    </p>
  </div>
);

// Info Cards Component
const InfoCards = () => (
  <div className="grid md:grid-cols-2 gap-6">
    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-6 border border-blue-100">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-blue-200 rounded-lg">
          <Brain className="w-6 h-6 text-blue-700" />
        </div>
        <h3 className="font-bold text-blue-900 text-lg">AI Technology</h3>
      </div>
      <ul className="space-y-2 text-sm text-blue-800">
        <li className="flex items-start gap-2">
          <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span>MobileNetV2 deep learning architecture</span>
        </li>
        <li className="flex items-start gap-2">
          <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span>Trained on 5,863 chest X-ray images</span>
        </li>
        <li className="flex items-start gap-2">
          <ChevronRight className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span>Transfer learning from ImageNet dataset</span>
        </li>
      </ul>
    </div>

    <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-6 border border-green-100">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-green-200 rounded-lg">
          <TrendingUp className="w-6 h-6 text-green-700" />
        </div>
        <h3 className="font-bold text-green-900 text-lg">
          Performance Metrics
        </h3>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white/50 rounded-lg p-3">
          <p className="text-xs text-green-600 mb-1">Accuracy</p>
          <p className="text-2xl font-bold text-green-900">89%</p>
        </div>
        <div className="bg-white/50 rounded-lg p-3">
          <p className="text-xs text-green-600 mb-1">Precision</p>
          <p className="text-2xl font-bold text-green-900">89%</p>
        </div>
        <div className="bg-white/50 rounded-lg p-3">
          <p className="text-xs text-green-600 mb-1">Recall</p>
          <p className="text-2xl font-bold text-green-900">94%</p>
        </div>
        <div className="bg-white/50 rounded-lg p-3">
          <p className="text-xs text-green-600 mb-1">F1-Score</p>
          <p className="text-2xl font-bold text-green-900">89%</p>
        </div>
      </div>
    </div>
  </div>
);

// Footer Component
const Footer = () => (
  <footer className="bg-gray-900 text-white mt-20">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <Activity className="w-6 h-6 text-blue-400" />
          <span className="font-semibold">AI Pneumonia Detector</span>
        </div>
        <p className="text-gray-400 text-sm text-center">
          Final Year Project - Medical Image Processing | Built with React,
          TensorFlow & MobileNetV2
        </p>
      </div>
    </div>
  </footer>
);

// Main App Component
function LandingPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];

    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Please select a valid image file");
        return;
      }

      if (file.size > 10 * 1024 * 1024) {
        setError("File size must be less than 10MB");
        return;
      }

      setSelectedFile(file);
      setError(null);
      setResult(null);

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || "Analysis failed");
      }
    } catch (err) {
      setError("Failed to analyze image. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      <Header />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {error && (
          <div className="mb-8">
            <ErrorMessage error={error} onDismiss={() => setError(null)} />
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          <UploadSection
            preview={preview}
            selectedFile={selectedFile}
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
            onReset={handleReset}
            loading={loading}
          />

          <div>
            {result ? (
              <ResultsSection result={result} />
            ) : (
              <ResultsPlaceholder />
            )}
          </div>
        </div>

        <InfoCards />
      </main>

      <Footer />
    </div>
  );
}

export default LandingPage;
