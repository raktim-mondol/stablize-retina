"""
Flask API for Retina Video Stabilization
"""
import os
import sys
import uuid
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Add parent directory to path to import retina_stabilizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.video_processor import VideoProcessor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Job storage (in-memory for demo)
jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available model configurations"""
    return jsonify({
        'models': [
            {
                'id': 'neural',
                'name': 'Neural (Best Quality)',
                'description': 'RAFT optical flow + U-Net vessel segmentation',
                'recommended': True
            },
            {
                'id': 'classical',
                'name': 'Classical (Faster)',
                'description': 'Frangi filter based approach, no GPU required',
                'recommended': False
            }
        ]
    })


@app.route('/api/stabilize', methods=['POST'])
def stabilize_video():
    """Upload and start video stabilization"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Get model selection
    model_type = request.form.get('model', 'neural')

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    ext = file.filename.rsplit('.', 1)[1].lower()
    input_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_input.{ext}')
    output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_output.mp4')

    file.save(input_path)

    # Initialize job
    jobs[job_id] = {
        'status': 'processing',
        'progress': 0,
        'stage': 'Initializing',
        'input_path': input_path,
        'output_path': output_path,
        'model': model_type,
        'metrics': None,
        'benchmark': None,
        'error': None
    }

    # Start processing in background thread
    processor = VideoProcessor()
    thread = threading.Thread(
        target=processor.process,
        args=(job_id, jobs, input_path, output_path, model_type)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'processing'})


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get job processing status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'stage': job['stage'],
        'error': job['error']
    })


@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Get stabilization results"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if job['status'] == 'processing':
        return jsonify({'error': 'Job still processing'}), 202

    if job['status'] == 'failed':
        return jsonify({'error': job['error']}), 500

    return jsonify({
        'status': 'completed',
        'video_url': f'/api/video/{job_id}/output',
        'original_url': f'/api/video/{job_id}/input',
        'metrics': job['metrics'],
        'benchmark': job['benchmark']
    })


@app.route('/api/video/<job_id>/<video_type>', methods=['GET'])
def get_video(job_id, video_type):
    """Stream video file"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if video_type == 'input':
        path = job['input_path']
    elif video_type == 'output':
        path = job['output_path']
    else:
        return jsonify({'error': 'Invalid video type'}), 400

    if not os.path.exists(path):
        return jsonify({'error': 'Video not found'}), 404

    return send_file(path, mimetype='video/mp4')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
