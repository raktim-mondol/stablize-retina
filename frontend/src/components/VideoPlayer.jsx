import React from 'react';

function VideoPlayer({ originalUrl, stabilizedUrl }) {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Video Comparison</h2>

      <div className="grid md:grid-cols-2 gap-4">
        {/* Original Video */}
        <div>
          <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-2">
            Original
          </h3>
          <div className="video-container">
            <video
              src={originalUrl}
              controls
              className="w-full rounded-lg"
            />
          </div>
        </div>

        {/* Stabilized Video */}
        <div>
          <h3 className="text-sm text-gray-400 uppercase tracking-wider mb-2">
            Stabilized
          </h3>
          <div className="video-container">
            <video
              src={stabilizedUrl}
              controls
              className="w-full rounded-lg"
            />
          </div>
        </div>
      </div>

      <p className="text-gray-400 text-sm mt-4 text-center">
        Right-click on stabilized video to download
      </p>
    </div>
  );
}

export default VideoPlayer;
