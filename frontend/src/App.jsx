import React, { useState, useEffect } from 'react';
import axios from 'axios';
import VideoUploader from './components/VideoUploader';
import ModelSelector from './components/ModelSelector';
import ProgressBar from './components/ProgressBar';
import VideoPlayer from './components/VideoPlayer';
import MetricsPanel from './components/MetricsPanel';
import BenchmarkPanel from './components/BenchmarkPanel';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('neural');
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Fetch available models on mount
  useEffect(() => {
    axios.get(`${API_URL}/models`)
      .then(res => {
        setModels(res.data.models);
        const recommended = res.data.models.find(m => m.recommended);
        if (recommended) setSelectedModel(recommended.id);
      })
      .catch(err => console.error('Failed to fetch models:', err));
  }, []);

  // Poll for status when job is processing
  useEffect(() => {
    if (!jobId || status?.status === 'completed' || status?.status === 'failed') {
      return;
    }

    const interval = setInterval(() => {
      axios.get(`${API_URL}/status/${jobId}`)
        .then(res => {
          setStatus(res.data);

          if (res.data.status === 'completed') {
            // Fetch results
            axios.get(`${API_URL}/result/${jobId}`)
              .then(resultRes => {
                setResult(resultRes.data);
              })
              .catch(err => setError(err.message));
          } else if (res.data.status === 'failed') {
            setError(res.data.error);
          }
        })
        .catch(err => setError(err.message));
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, status]);

  const handleUpload = async () => {
    if (!file) return;

    setError(null);
    setResult(null);
    setStatus(null);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('model', selectedModel);

    try {
      const res = await axios.post(`${API_URL}/stabilize`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setJobId(res.data.job_id);
      setStatus({ status: 'processing', progress: 0, stage: 'Starting' });
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    }
  };

  const handleReset = () => {
    setFile(null);
    setJobId(null);
    setStatus(null);
    setResult(null);
    setError(null);
  };

  const isProcessing = status?.status === 'processing';
  const isComplete = status?.status === 'completed';

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Header */}
      <header className="border-b border-dark-700 py-6">
        <div className="max-w-7xl mx-auto px-6">
          <h1 className="text-3xl font-bold tracking-tight">
            Retina Video Stabilizer
          </h1>
          <p className="text-gray-400 mt-1">
            Advanced retinal video stabilization using deep learning
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Upload Section */}
        {!isProcessing && !isComplete && (
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <VideoUploader file={file} setFile={setFile} />
              <ModelSelector
                models={models}
                selected={selectedModel}
                setSelected={setSelectedModel}
              />
            </div>

            {file && (
              <div className="flex justify-center">
                <button
                  onClick={handleUpload}
                  className="btn-primary text-lg px-8 py-4"
                >
                  Stabilize Video
                </button>
              </div>
            )}
          </div>
        )}

        {/* Processing Status */}
        {isProcessing && (
          <div className="card max-w-2xl mx-auto">
            <h2 className="text-xl font-semibold mb-4">Processing Video</h2>
            <ProgressBar
              progress={status.progress}
              stage={status.stage}
            />
            <p className="text-gray-400 text-sm mt-4 text-center">
              This may take a few minutes depending on video length
            </p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="card max-w-2xl mx-auto border-red-500/50 bg-red-500/10">
            <h2 className="text-xl font-semibold text-red-400 mb-2">Error</h2>
            <p className="text-gray-300">{error}</p>
            <button onClick={handleReset} className="btn-secondary mt-4">
              Try Again
            </button>
          </div>
        )}

        {/* Results Section */}
        {isComplete && result && (
          <div className="space-y-6">
            {/* Video Comparison */}
            <VideoPlayer
              originalUrl={`${API_URL.replace('/api', '')}${result.original_url}`}
              stabilizedUrl={`${API_URL.replace('/api', '')}${result.video_url}`}
            />

            {/* Metrics Grid */}
            <div className="grid lg:grid-cols-2 gap-6">
              <MetricsPanel metrics={result.metrics} />
              <BenchmarkPanel benchmark={result.benchmark} />
            </div>

            {/* Reset Button */}
            <div className="flex justify-center pt-4">
              <button onClick={handleReset} className="btn-secondary">
                Process Another Video
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-dark-700 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-6 text-center text-gray-400 text-sm">
          Retina Video Stabilization System
        </div>
      </footer>
    </div>
  );
}

export default App;
