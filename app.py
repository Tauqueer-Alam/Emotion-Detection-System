import cv2
import base64
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# Load the face cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def process_frame(frame):
    colors = {
        'angry': (0, 0, 255),       # Red
        'disgust': (0, 255, 0),     # Green
        'fear': (128, 0, 128),      # Purple
        'happy': (0, 255, 255),     # Yellow
        'sad': (255, 0, 0),         # Blue
        'surprise': (255, 255, 0),  # Cyan
        'neutral': (255, 255, 255)  # White
    }
    
    try:
        # Instead of Haar Cascades making a static crop, let DeepFace handle the full frame 
        # and locate the faces naturally. It improves prediction accuracy drastically.
        # We enforce detection so it filters out false positives.
        results = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=True, 
            detector_backend='mtcnn' # Changed to a Deep Learning (CNN) face detector for better accuracy
        )
        
        # DeepFace returns a list of dictionaries if multiple faces are found
        if not isinstance(results, list):
            results = [results]
            
        for analysis in results:
            emotion = analysis['dominant_emotion']
            score = analysis['emotion'][emotion]
            
            # DeepFace also returns the bounding box it found
            region = analysis['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Setup colors
            color = colors.get(emotion, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw label
            text = f"{emotion.capitalize()} ({score:.1f}%)"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, cv2.FILLED)
            
            # Check luminance for text color readability
            bg_luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if bg_luminance > 128 else (255, 255, 255)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
    except Exception as e:
        # DeepFace throws an exception if no face is found (enforce_detection=True)
        pass
        
    return frame

def gen_frames():
    camera = cv2.VideoCapture(0)
    # Reduce buffer size to minimize lag during streaming
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process the frame to detect and draw emotions
            frame = process_frame(frame)
            
            # Encode frame to JPEG format for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the output frame to the video stream via boundary protocol
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return a multipart response containing chunks of frames
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame_api', methods=['POST'])
def process_frame_api():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Extract base64 image data
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 to OpenCV image
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process the frame
        processed_frame = process_frame(frame)

        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'image': 'data:image/jpeg;base64,' + encoded_img})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Added ssl_context='adhoc' so mobile browsers permit camera recording (HTTPS required for mobile cams)
    app.run(host='0.0.0.0', debug=True, port=5000, ssl_context='adhoc')
