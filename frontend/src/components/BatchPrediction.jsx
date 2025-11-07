import { useState, useRef } from 'react';
import { Upload, X, CheckCircle, AlertCircle, Loader2, Layers, Download, FileText } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:5000';

export default function BatchPrediction() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files || []);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
      setError('Please select valid image files');
      return;
    }

    if (imageFiles.length !== files.length) {
      setError('Some files were skipped (only image files are accepted)');
    } else {
      setError(null);
    }

    setSelectedFiles(imageFiles);
    setResults(null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files || []);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
      setError('Please select valid image files');
      return;
    }

    if (imageFiles.length !== files.length) {
      setError('Some files were skipped (only image files are accepted)');
    } else {
      setError(null);
    }

    setSelectedFiles(imageFiles);
    setResults(null);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append('images', file);
      });

      const response = await axios.post(`${API_URL}/api/batch-predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setResults(response.data.results);
      } else {
        setError(response.data.error || 'Analysis failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to connect to server. Make sure the backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFiles([]);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const downloadResults = () => {
    if (!results) return;

    const csvContent = [
      ['Filename', 'Prediction', 'Confidence (%)'],
      ...results.map(r => [
        r.filename,
        r.prediction || 'Error',
        r.confidence || 'N/A'
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_analysis_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getStats = () => {
    if (!results) return null;

    const successful = results.filter(r => !r.error);
    const pneumoniaCount = successful.filter(r => r.prediction === 'PNEUMONIA').length;
    const normalCount = successful.filter(r => r.prediction === 'NORMAL').length;
    const errorCount = results.filter(r => r.error).length;

    return { successful: successful.length, pneumoniaCount, normalCount, errorCount };
  };

  const stats = getStats();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 animate-fade-in">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Batch Image Analysis
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Upload multiple chest X-ray images for simultaneous pneumonia detection analysis
        </p>
      </div>

      {/* Upload Section */}
      <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-200 mb-8">
        <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
          <Layers className="h-5 w-5 text-blue-600" />
          Upload Multiple Images
        </h2>

        {/* Drag and Drop Area */}
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50/50 transition-all mb-6"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />

          <div className="space-y-4">
            <div className="flex justify-center">
              <div className="bg-blue-100 p-4 rounded-full">
                <Upload className="h-10 w-10 text-blue-600" />
              </div>
            </div>
            <div>
              <p className="text-lg font-medium text-gray-900 mb-2">
                Click to upload or drag and drop
              </p>
              <p className="text-sm text-gray-500">
                Multiple PNG, JPG or JPEG files
              </p>
            </div>
          </div>
        </div>

        {/* Selected Files List */}
        {selectedFiles.length > 0 && (
          <div className="mb-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-semibold text-gray-900">
                Selected Files ({selectedFiles.length})
              </h3>
              <button
                onClick={handleReset}
                disabled={loading}
                className="text-sm text-red-600 hover:text-red-700 font-medium disabled:opacity-50"
              >
                Clear All
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-64 overflow-y-auto">
              {selectedFiles.map((file, index) => (
                <div
                  key={index}
                  className="bg-gray-50 rounded-lg p-3 flex items-center gap-3 group hover:bg-gray-100 transition-colors"
                >
                  <div className="bg-blue-100 p-2 rounded">
                    <FileText className="h-4 w-4 text-blue-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(index);
                    }}
                    disabled={loading}
                    className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-700 transition-opacity disabled:opacity-50"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action Button */}
        <button
          onClick={handleAnalyze}
          disabled={selectedFiles.length === 0 || loading}
          className="w-full bg-blue-600 text-white px-6 py-3 rounded-xl font-semibold hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              Analyzing {selectedFiles.length} images...
            </>
          ) : (
            <>
              <Layers className="h-5 w-5" />
              Analyze {selectedFiles.length > 0 ? selectedFiles.length : ''} Images
            </>
          )}
        </button>

        {/* Error Message */}
        {error && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-900">Error</p>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        )}
      </div>

      {/* Results Section */}
      {(results || loading) && (
        <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-200">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">
              Analysis Results
            </h2>
            {results && (
              <button
                onClick={downloadResults}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                <Download className="h-4 w-4" />
                Download CSV
              </button>
            )}
          </div>

          {loading && (
            <div className="text-center py-12">
              <Loader2 className="h-12 w-12 text-blue-600 animate-spin mx-auto mb-4" />
              <p className="text-gray-600 font-medium">
                Analyzing {selectedFiles.length} images...
              </p>
              <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
            </div>
          )}

          {results && stats && (
            <div className="space-y-6 animate-fade-in">
              {/* Summary Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
                  <div className="text-2xl font-bold text-blue-900">{results.length}</div>
                  <div className="text-sm text-blue-700">Total Analyzed</div>
                </div>
                <div className="bg-emerald-50 rounded-xl p-4 border border-emerald-200">
                  <div className="text-2xl font-bold text-emerald-900">{stats.normalCount}</div>
                  <div className="text-sm text-emerald-700">Normal</div>
                </div>
                <div className="bg-red-50 rounded-xl p-4 border border-red-200">
                  <div className="text-2xl font-bold text-red-900">{stats.pneumoniaCount}</div>
                  <div className="text-sm text-red-700">Pneumonia</div>
                </div>
                <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
                  <div className="text-2xl font-bold text-gray-900">{stats.errorCount}</div>
                  <div className="text-sm text-gray-700">Errors</div>
                </div>
              </div>

              {/* Results Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50 border-b-2 border-gray-200">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">
                        #
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">
                        Filename
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">
                        Prediction
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">
                        Confidence
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {results.map((result, index) => (
                      <tr key={index} className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-3 text-sm text-gray-600">
                          {index + 1}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 font-medium">
                          {result.filename}
                        </td>
                        <td className="px-4 py-3">
                          {result.error ? (
                            <span className="text-sm text-gray-500">-</span>
                          ) : (
                            <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                              result.prediction === 'PNEUMONIA'
                                ? 'bg-red-100 text-red-800'
                                : 'bg-emerald-100 text-emerald-800'
                            }`}>
                              {result.prediction}
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {result.error ? (
                            <span className="text-gray-500">-</span>
                          ) : (
                            <div className="flex items-center gap-2">
                              <div className="flex-1 bg-gray-200 rounded-full h-2 min-w-[80px]">
                                <div
                                  className={`h-full rounded-full ${
                                    result.prediction === 'PNEUMONIA'
                                      ? 'bg-red-500'
                                      : 'bg-emerald-500'
                                  }`}
                                  style={{ width: `${result.confidence}%` }}
                                ></div>
                              </div>
                              <span className="text-gray-900 font-medium min-w-[45px]">
                                {result.confidence}%
                              </span>
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {result.error ? (
                            <div className="flex items-center gap-2 text-red-600">
                              <AlertCircle className="h-4 w-4" />
                              <span className="text-xs">Error</span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-2 text-emerald-600">
                              <CheckCircle className="h-4 w-4" />
                              <span className="text-xs">Success</span>
                            </div>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Disclaimer */}
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-xs text-yellow-800 leading-relaxed">
                  <strong>Disclaimer:</strong> These AI analyses are for informational purposes only 
                  and should not replace professional medical diagnosis. Please consult with qualified 
                  healthcare providers for medical advice.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Info Card */}
      {!results && !loading && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
          <h3 className="font-semibold text-blue-900 mb-3">Batch Processing Tips:</h3>
          <ul className="space-y-2 text-sm text-blue-800">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>Upload multiple chest X-ray images at once for efficient analysis</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>All results can be downloaded as a CSV file for record-keeping</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>Supported formats: JPEG, JPG, PNG</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>Processing time increases with the number of images</span>
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
