import React, { useRef } from 'react';

function VideoUploader({ file, setFile }) {
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="card">
      <h2 className="text-lg font-semibold mb-4">Upload Video</h2>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={handleClick}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200
          ${file
            ? 'border-white bg-dark-700'
            : 'border-dark-600 hover:border-gray-400'
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          onChange={handleChange}
          className="hidden"
        />

        {file ? (
          <div>
            <div className="text-white font-medium mb-1">{file.name}</div>
            <div className="text-gray-400 text-sm">{formatSize(file.size)}</div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setFile(null);
              }}
              className="mt-3 text-sm text-gray-400 hover:text-white"
            >
              Remove
            </button>
          </div>
        ) : (
          <div>
            <div className="text-4xl mb-3">
              <svg className="w-12 h-12 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="text-gray-300 mb-1">
              Drop video here or click to browse
            </div>
            <div className="text-gray-400 text-sm">
              MP4, AVI, MOV, MKV, WebM
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoUploader;
