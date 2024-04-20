from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask import Flask, render_template, url_for

from flask import jsonify
import json

from DrownDetector import detectDrowning
  

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov'}

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')


@app.route('/livecamera')
def livecamera():
    return render_template('livecamera.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        video_file = request.files['video']

        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        if video_file and allowed_file(video_file.filename):
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)

            print("Video path:", video_path) 
            
           
            # Call detectDrowning and capture the swimming status
            swimming_status = detectDrowning(video_file.filename)
            print("Swimming_Status:", swimming_status) 

            # Determine if drowning based on swimming status
            is_drowning = True if swimming_status == 'drowning' else False
            print("is_drowning:", is_drowning)       

            # Send the result as a JSON response, including swimming status
            return jsonify({'success': True, 'video_url': video_path, 'is_drowning': is_drowning}), 200
       

            
        else:
            return jsonify({'success': False, 'error': 'Invalid file format'}), 400
    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)

