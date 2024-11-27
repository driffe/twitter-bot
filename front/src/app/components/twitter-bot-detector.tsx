'use client';
import { useState } from 'react';
import { Search } from 'lucide-react';

interface AnalysisResult {
  bot_status: boolean;
  error?: string;
}

export function TwitterBotDetector() {
  const [username, setUsername] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const analyzeUser = async () => {
    if (!username.trim()) {
      setError('Please enter a Twitter username');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      // 요청 전 로깅
      console.log('Attempting to send request to server...');
      
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        mode: 'cors',
        cache: 'no-cache',
        credentials: 'omit',
        body: JSON.stringify({
          screen_name: username
        })
      });

      console.log('Response received:', response);
      
      const data = await response.json();
      console.log('Data received:', data);
      
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      // 더 자세한 에러 로깅
      if (err instanceof Error) {
        console.error('Error details:', {
          message: err.message,
          name: err.name,
          stack: err.stack
        });
      } else {
        console.error('Unknown error:', err);
      }
      setError('Failed to analyze user. Please try again.');
    } finally {
      setLoading(false);
    }
};

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-3xl space-y-8">
        <h1 className="text-5xl font-bold text-center bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
          Twitter Bot Detector
        </h1>

        <div className="bg-gray-800/50 p-6 rounded-2xl backdrop-blur-sm shadow-xl">
          <div className="flex gap-3">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter Twitter Username"
              className="flex-1 bg-gray-700/50 text-white rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-blue-500 transition-all"
            />
            <button
              onClick={analyzeUser}
              disabled={loading}
              className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-xl flex items-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Search className="w-5 h-5" />
              )}
              Analyze
            </button>
          </div>

          {error && (
            <div className="mt-4 text-red-400 text-sm">
              {error}
            </div>
          )}
        </div>

        {result && !loading && (
          <div className="bg-gray-800/50 p-6 rounded-2xl backdrop-blur-sm shadow-xl space-y-6 animate-fadeIn">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-semibold text-white">Analysis Results</h2>
              <div className={`px-4 py-2 rounded-lg ${
                result.bot_status 
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-green-500/20 text-green-400'
              }`}>
                {result.bot_status ? 'Bot Account' : 'Human Account'}
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-700/50 p-4 rounded-xl">
                <div className="text-gray-400 mb-1">Username</div>
                <div className="text-xl text-white">@{username}</div>
              </div>
              
              <div className="bg-gray-700/50 p-4 rounded-xl">
                <div className="text-gray-400 mb-1">Confidence Level</div>
                <div className="text-xl text-white">
                  {result.bot_status ? 'High' : 'Medium'}
                </div>
              </div>
            </div>

            <div className="bg-gray-700/50 p-4 rounded-xl">
              <div className="text-gray-400 mb-2">Analysis Summary</div>
              <p className="text-white">
                {result.bot_status 
                  ? "This account shows patterns typically associated with automated behavior. It may be used for automated posting, content amplification, or other automated activities."
                  : "This account displays characteristics consistent with human behavior. The posting patterns, engagement levels, and account characteristics align with typical human usage."}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}