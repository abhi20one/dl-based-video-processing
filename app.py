import os
import time
import threading
import queue
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for, flash
from ultralytics import YOLO
import face_recognition
import shutil
import json
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import time
import subprocess
from typing import Optional
import cv2
import yt_dlp
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
from typing import Optional
import yt_dlp

app = Flask(__name__)

# fileUpload code part
app.config['SECRET_KEY']='supersecretkey'
app.config['UPLOAD_FOLDER']='videos'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

IMAGE_FOLDER = 'faces'  # Directory containing images

MODEL_DIR = "models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
default_model = 'yolov8n.pt'

video_source_name = 0
# Global variables for pausing and skipping frames
is_paused = True
last_frame = None  # Store last frame for pausing
skip_frames = 0  # Number of frames to skip for fast forward/rewind, initially set 0

seek_frames = 0
seek_time = 0  # Time in seconds to seek to

human_detection_enabled = True  # Default is ON
auto_detect = True  # Default to True, i.e., auto-detect is on
auto_save_faces = False  # Default to True, i.e., auto-save faces is on

frame_rate = 30 #default fps set

# Store ROI coordinates globally
roi_coords = [0, 0, 1200, 675]  # Default ROI: (x1, y1, x2, y2)

# Directory to store detected face images
MATCHED_FACES_DIR = "matched_faces"
MATCHED_FACES_ZOOM_DIR = "matched_faces_zoom"
if not os.path.exists(MATCHED_FACES_DIR):
    os.makedirs(MATCHED_FACES_DIR)
if not os.path.exists(MATCHED_FACES_ZOOM_DIR):
    os.makedirs(MATCHED_FACES_ZOOM_DIR)

matched_faces = []  # Store {"name": "Person", "img_url": "/matched_faces/person.jpg"}

# Face recognition queue
face_queue = queue.Queue()

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET',"POST"])
def index():
    video_files = [f for f in os.listdir('videos') if f.endswith(('.mp4', '.avi', '.mkv'))]
    images = os.listdir(IMAGE_FOLDER)  # List all files in the "faces" directory
    images = [img for img in images if img.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]  # Filter images
    image_data = [{'filename': img, 'name': os.path.splitext(img)[0]} for img in images]

    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        upload_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config['UPLOAD_FOLDER'],
            filename
        )
        file.save(upload_path)

        flash(f"File '{filename}' uploaded successfully!", "success")
        return redirect(url_for('index'))  # ✅ redirect instead of returning text

    return render_template('index.html', form=form, video_files=video_files, model_files=model_files, default_model=default_model, images=image_data)

@app.route('/set_roi', methods=['POST'])
def set_roi():
    """Receive and validate ROI coordinates from the frontend."""
    global roi_coords
    data = request.json

    x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']

    # Ensure x2 > x1 and y2 > y1 (swap if needed)
    if x2 <= x1:
        x2 = x1 + 1  # Ensure at least 1px width
    if y2 <= y1:
        y2 = y1 + 1  # Ensure at least 1px height

    roi_coords = [x1, y1, x2, y2]
    return jsonify({'message': 'ROI updated', 'roi': roi_coords})

IMAGE_FOLDER = 'faces'
CACHE_DIR = 'cached_face_encodings'
ENCODINGS_FILE = os.path.join(CACHE_DIR, 'face_encodings.npy')
NAMES_FILE = os.path.join(CACHE_DIR, 'face_names.json')
META_FILE = os.path.join(CACHE_DIR, 'cache_meta.json')

# Load known faces
known_faces = []
known_names = []

def process_face(file_path):
    """Loads an image and returns encoding tuple or SKIPPED marker."""
    try:
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            name = os.path.splitext(os.path.basename(file_path))[0]
            return ("ENCODED", encodings[0], name, file_path)
        else:
            print(f"[SKIP] No face found in {file_path}")
            return ("SKIPPED", None, None, file_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return ("SKIPPED", None, None, file_path)

def load_known_faces():
    global known_faces, known_names

    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load previous cache
    cached_encodings = []
    cached_names = []
    cached_meta = {}

    if os.path.exists(ENCODINGS_FILE) and os.path.exists(NAMES_FILE) and os.path.exists(META_FILE):
        try:
            cached_encodings = np.load(ENCODINGS_FILE, allow_pickle=True).tolist()
            with open(NAMES_FILE, 'r') as f:
                cached_names = json.load(f)
            with open(META_FILE, 'r') as f:
                cached_meta = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load old cache: {e}")
            cached_encodings = []
            cached_names = []
            cached_meta = {}

    all_files = [
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    unchanged_encodings = []
    unchanged_names = []
    new_files = []
    new_meta = {}

    print(f"[INFO] Checking {len(all_files)} image(s)...")

    for fpath in all_files:
        fname = os.path.basename(fpath)
        name = os.path.splitext(fname)[0]
        last_modified = os.path.getmtime(fpath)

        if fname in cached_meta:
            cached_time = cached_meta[fname]
            if isinstance(cached_time, float) and abs(cached_time - last_modified) < 1:
                # Valid face encoding exists
                if name in cached_names:
                    try:
                        idx = cached_names.index(name)
                        unchanged_encodings.append(cached_encodings[idx])
                        unchanged_names.append(name)
                        new_meta[fname] = cached_time
                    except ValueError:
                        pass  # Skip
                else:
                    # Previously skipped — still unchanged
                    print(f"[SKIP-CACHE] {fname} was previously skipped and hasn't changed.")
                    new_meta[fname] = cached_time
            else:
                # File has changed — reprocess
                new_files.append(fpath)
        else:
            # New file
            new_files.append(fpath)

    print(f"[INFO] {len(unchanged_names)} face(s) loaded from cache.")
    print(f"[INFO] {len(new_files)} new/changed image(s) to encode...")

    start = time.time()
    new_encodings = []
    new_names = {}

    if new_files:
        with Pool(cpu_count()) as pool:
            for result in tqdm(pool.imap(process_face, new_files), total=len(new_files), desc="Encoding new faces"):
                if result:
                    status, enc, name, path = result
                    fname = os.path.basename(path)
                    if status == "ENCODED":
                        new_encodings.append(enc)
                        new_names[name] = os.path.getmtime(path)
                        unchanged_encodings.append(enc)
                        unchanged_names.append(name)
                        new_meta[fname] = os.path.getmtime(path)
                    elif status == "SKIPPED":
                        new_meta[fname] = os.path.getmtime(path)  # cache skip

    # Merge new meta with retained meta
    new_meta.update(new_names)

    # Save final data
    known_faces = unchanged_encodings
    known_names = unchanged_names

    np.save(ENCODINGS_FILE, known_faces)
    with open(NAMES_FILE, 'w') as f:
        json.dump(known_names, f)
    with open(META_FILE, 'w') as f:
        json.dump(new_meta, f)

    print(f"\n✅ Total: {len(known_faces)} face(s) ready in {time.time() - start:.2f} seconds.\n")

def face_recognition_worker():
    """Background thread for face recognition."""
    while True:
        if face_queue.empty():
            time.sleep(0.01)
            continue

        frame_data = face_queue.get()
        if frame_data is None:  # Stop signal
            break

        process_faces(*frame_data)

# Directory where unknown faces are stored
UNKNOWN_FACES_DIR = "faces"

def process_faces(person_crop, xmin, ymin):
    """Processes detected faces and updates unknown face list in real-time."""
    global matched_faces, known_faces, known_names

    rgb_face = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_face, model="hog") 
    face_encodings = face_recognition.face_encodings(rgb_face, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
        else:
            if auto_save_faces:
                # Save whole person crop for unknown faces
                if not os.path.exists(UNKNOWN_FACES_DIR):
                    os.makedirs(UNKNOWN_FACES_DIR)

                unknown_index = 1
                while os.path.exists(os.path.join(UNKNOWN_FACES_DIR, f"unknown_{unknown_index}.jpg")):
                    unknown_index += 1

                face_filename = f"unknown_{unknown_index}.jpg"  #  Correct filename format
                face_path = os.path.join(UNKNOWN_FACES_DIR, face_filename)

                cv2.imwrite(face_path, person_crop)  # Save entire person_crop
                print(f"New unknown face saved: {face_filename}")

                #  **Dynamically Load the New Unknown Face**
                new_face_image = face_recognition.load_image_file(face_path)
                new_face_encoding = face_recognition.face_encodings(new_face_image)
            
                if new_face_encoding:
                    known_faces.append(new_face_encoding[0])  # Add to known faces
                    known_names.append(face_filename.replace(".jpg", ""))  #  Avoid extra ".jpg"
                    print(f"New unknown face loaded into the system: {face_filename}")

        #  Avoid appending ".jpg" again if it's already there
        if not name.endswith(".jpg"):
            face_filename = f"{name}.jpg"
        else:
            face_filename = name

        face_path = os.path.join(MATCHED_FACES_DIR, face_filename)
        face_path_zoom = os.path.join(MATCHED_FACES_ZOOM_DIR, face_filename)

        if name != "Unknown" and not any(f["name"] == name for f in matched_faces):
            # Save whole person crop for matched faces
            face_crop = person_crop[top:bottom, left:right]
            cv2.imwrite(face_path_zoom, face_crop)
            cv2.imwrite(face_path, person_crop)
            matched_faces.append({
                "name": name, 
                "video_source_name": video_source_name,
                "img_url1": f"/matched_faces/{face_filename}",
                "img_url2": f"/matched_faces_zoom/{face_filename}",
                "img_url3": f"/faces/{face_filename}", 
                "time_stamp": time_text})

@app.route("/clear_matched_faces", methods=["POST"])
def clear_matched_faces():
    global matched_faces
    matched_faces.clear()

    # Delete all files in MATCHED_FACES_DIR
    if os.path.exists(MATCHED_FACES_DIR):
        shutil.rmtree(MATCHED_FACES_DIR)
        os.makedirs(MATCHED_FACES_DIR)  # Recreate the empty directory

    if os.path.exists(MATCHED_FACES_ZOOM_DIR):
        shutil.rmtree(MATCHED_FACES_ZOOM_DIR)
        os.makedirs(MATCHED_FACES_ZOOM_DIR)  # Recreate the empty directory

    return jsonify({"message": "Matched faces cleared!"})

@app.route('/seek', methods=['POST'])
def handle_seek():
    """Handle frame skipping (fast forward or rewind)."""
    global skip_frames
    data = request.get_json()

    action = data.get('action', '')
    frames = data.get('frames', 0)

    if action == 'fastforward':
        skip_frames += frames
    elif action == 'rewind' :
        skip_frames -= frames

    return jsonify({"status": "success", "skip_frames": skip_frames})

@app.route('/seek', methods=['POST'])
def seek():
    """Seek to a specific timestamp in the video."""
    global seek_frames
    data = request.get_json()

    target_time = data.get('time', 0)  # Time in seconds
    # print(f"seek video frame rate: {frame_rate}")
    seek_frames = int(target_time * frame_rate)  # Convert time to frame number
    return jsonify({"status": "success", "seek_time": target_time})

def generate_frames(video_source, model_path):
    """Generates video frames and detects objects."""
    global is_paused, last_frame, auto_detect, skip_frames, frame_rate, seek_frames

    model = YOLO(model_path, task='detect')
    labels = model.names

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    frame_skip = 5        #number of frames skipped for face detection
    frame_skip_count = 0
    yolo_frame_count = 0  # Counter to process every 5th frame
    yolo_frame_skip = 1   # frame skip for yolo detection 

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30  # Avoid division by zero
    # print(f"video frame rate: {frame_rate}")
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0  # Total duration in seconds

    # prev_frame_time = 0  # Start time for FPS calculation
    new_frame_time = 0
    prev_frame_time = time.time()
    frame_count = 0
    fps = 0

    current_frame_position = 0  # Track the current frame position

    yolo_frame_count = 0  # Counter to process every 5th frame

    target_frame_time = 1 / frame_rate
    last_time = time.time()

    while True:
        if is_paused and last_frame is not None:
            ret, buffer = cv2.imencode('.jpg', last_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
            continue

        if seek_frames != 0:
            current_frame_position = seek_frames  # Seek to frame based on skip_frames
            current_frame_position = max(0, min(current_frame_position, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_position)
            seek_frames = 0  # Reset after seeking

        if skip_frames != 0:
            # Adjust frame position based on fast-forward or rewind
            if skip_frames > 0:
                current_frame_position += skip_frames
            elif skip_frames < 0:
                current_frame_position += skip_frames
            # Ensure the frame position stays within bounds
            current_frame_position = max(0, min(current_frame_position, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_position)  # Move to the new frame position
            skip_frames = 0  # Reset skip frames after moving
        
        ret, frame = cap.read()
        if not ret:
            # print('Reached the end of the video file.')
            break

        yolo_frame_count += 1    
        frame_skip_count += 1
        object_count = 0

        # Get the actual frame size
        frame_h, frame_w, _ = frame.shape  # Full-resolution video frame
        # Get frontend image size (assume fixed for now)
        display_w, display_h = 1200, 675  # Set based on frontend `img.width` (adjust if different)
        # Scale ROI from frontend coordinates to actual frame size
        scale_x = frame_w / display_w
        scale_y = frame_h / display_h
        x1, y1, x2, y2 = [int(roi_coords[i] * scale_x) if i % 2 == 0 else int(roi_coords[i] * scale_y) for i in range(4)]
        # Ensure ROI is within bounds
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
        # Extract and process the ROI
        roi = frame[y1:y2, x1:x2]

        if yolo_frame_count % yolo_frame_skip == 0:  # Process every 5th frame
            if human_detection_enabled:
        
                # Make a copy of the original frame before any modifications
                clean_frame = frame.copy()  
    
                results = model(roi, verbose=False)
                detections = results[0].boxes
            
                for i in range(len(detections)):
                    xyxy_tensor = detections[i].xyxy.cpu()
                    xyxy = xyxy_tensor.numpy().squeeze()
                    xmin, ymin, xmax, ymax = xyxy.astype(int)

                    classidx = int(detections[i].cls.item())
                    classname = labels[classidx]
                    conf = detections[i].conf.item()

                    if conf >= 0.5:
                        color = (0, 255, 255)

                        xmin += x1  # Adjust ROI back to full frame
                        xmax += x1
                        ymin += y1
                        ymax += y1

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    
                        label = f'{classname}: {int(conf * 100)}%'
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                        if classname == 'person':
                            object_count += 1

                            # Only auto-detect faces if the flag is enabled
                            if auto_detect and frame_skip_count % frame_skip == 0:
                                person_crop = clean_frame[ymin:ymax, xmin:xmax]
                                face_queue.put((person_crop, xmin, ymin))  # Send to face recognition worker

            # Draw the correct-sized ROI on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for ROI
        
            # Calculate FPS based on time difference between frames
            new_frame_time = time.time()
            frame_count += 1

            if new_frame_time - prev_frame_time >= 1:  # Update FPS every second
                fps = frame_count / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                frame_count = 0  # Reset frame count

            # Get current timestamp in the video
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert milliseconds to seconds

            # Format time as MM:SS
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                seconds = int(seconds % 60)
                return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

            global time_text
            time_text = f'Time: {format_time(current_time)} / {format_time(duration)}'
            fps_text = f'FPS: {fps:.2f}'
            people_text = f'People detected: {object_count}'

            # Display information on the frame
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, people_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, time_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            last_frame = frame.copy()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Sync with wall-clock
            elapsed = time.time() - last_time
            delay = target_frame_time - elapsed
            if delay > 0:
                time.sleep(delay)
            last_time = time.time()

    cap.release()

def extract_stream_url_python(url: str, cookies_path: str = None) -> Optional[str]:
    """
    Extracts a direct video/stream URL from YouTube, Twitch, Kick, or similar sites.
    Args:
        url (str): The video or stream page URL.
        cookies_path (str, optional): Path to browser cookies (if required for restricted streams).
    Returns:
        str | None: Direct stream URL if found, else None.
    """
    format_string = (
        "bestvideo[height=720][ext=mp4]/" #fix streaming resolution to 720p
        "best[height=720][ext=mp4]/"
        "best[protocol^=m3u8]/"
        "best"
    )

    ydl_opts = {
        "format": format_string,
        "noplaylist": True,
        "quiet": True,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/116.0.5845.96 Safari/537.36"
            ),
            "Client-ID": "kimne78kx3ncx6brgo4mv6wki5h1ko",  # Twitch public Client-ID
        },
    }

    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Direct URL
            if url := info.get("url"):
                return url

            # Progressive MP4 streams (video+audio)
            formats = info.get("formats") or []
            progressive = [
                f for f in formats
                if (f.get("ext") == "mp4" and f.get("vcodec") not in (None, "none")
                    and f.get("acodec") not in (None, "none") and f.get("url"))
            ]
            if progressive:
                progressive.sort(key=lambda f: f.get("height") or 0, reverse=True)
                return progressive[0]["url"]

            # Any other MP4 video-only streams
            mp4_video = [
                f for f in formats
                if f.get("ext") == "mp4" and f.get("vcodec") not in (None, "none") and f.get("url")
            ]
            if mp4_video:
                mp4_video.sort(key=lambda f: f.get("height") or 0, reverse=True)
                return mp4_video[0]["url"]

            # Fallback to any available URL
            any_url = [f for f in formats if f.get("url")]
            if any_url:
                any_url.sort(
                    key=lambda f: ((f.get("vcodec") not in (None, "none")) * 2) + (f.get("height") or 0),
                    reverse=True,
                )
                return any_url[0]["url"]

    except Exception as e:
        print(f"[yt-dlp Python API] extract failed: {e}")

    return None

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    global roi_coords  # Allow modification of global ROI coordinates

    video_source = request.args.get('source')
    selected_model = default_model
    print(default_model) 
    yt_url = request.args.get('vidurl')

    print(f"\n\nvideo_feed function is called")  
    print(f"video source: {video_source}")  
    print(f"selected model: {selected_model}")  
    print(f"yt url: {yt_url}")
    print("\n\n")  

    # Reset ROI to default whenever a new video starts
    roi_coords = [0, 0, 1200, 675]

    # If a URL is provided, use it as the video source
    direct_url = extract_stream_url_python(yt_url)
    if direct_url:
        video_source = direct_url
        print(f"Using stream URL: {direct_url}")  # <-- Print the direct stream URL
    elif video_source:
        # Check if the video source is a file or webcam
        if video_source == 'Webcam':
            video_source = 0  # Use webcam
        else:
            video_source = os.path.join('videos', video_source)
 

    # Use the selected model path for YOLO
    model_path = os.path.join(MODEL_DIR, selected_model)

    return Response(
        generate_frames(video_source, model_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/toggle_human_detection', methods=['POST'])
def toggle_human_detection():
    """Toggle human detection on/off."""
    global human_detection_enabled
    human_detection_enabled = not human_detection_enabled
    return jsonify({"human_detection_enabled": human_detection_enabled})

@app.route('/display_human_detection', methods=['POST'])
def display_human_detection():
    global human_detection_enabled
    return jsonify({"human_detection_enabled": human_detection_enabled})

@app.route('/toggle_auto_save_faces', methods=['POST'])
def toggle_auto_save_faces():
    """Toggle the auto-save faces setting on or off."""
    global auto_save_faces
    auto_save_faces = not auto_save_faces
    return jsonify({"auto_save_faces": auto_save_faces})

@app.route('/display_auto_save_faces', methods=['POST'])
def display_auto_save_faces():
    global auto_save_faces
    return jsonify({"auto_save_faces": auto_save_faces})

@app.route('/toggle_auto_detect', methods=['POST'])
def toggle_auto_detect():
    """Toggle the auto-detect setting on or off."""
    global auto_detect
    auto_detect = not auto_detect
    return jsonify({"auto_detect": auto_detect})

@app.route('/display_auto_detect', methods=['POST'])
def display_auto_detect():
    global auto_detect
    return jsonify({"auto_detect": auto_detect})

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({"paused": is_paused})

@app.route('/play_video', methods=['POST'])
def play_video():
    global is_paused
    if is_paused:  
        is_paused = False  # Resume playing
        return jsonify({"paused": False})  # Indicate video is now playing
    else:
        return jsonify({"message": "Already playing"})  # No action needed
 
@app.route('/display_pause', methods=['POST'])
def display_pause():
    global is_paused
    return jsonify({"paused": is_paused})

@app.route('/matched_faces')
def get_matched_faces():
    return jsonify(matched_faces)

@app.route('/matched_faces/<filename>')
def serve_matched_face(filename):
    return send_from_directory(MATCHED_FACES_DIR, filename)

@app.route('/matched_faces_zoom/<filename>')
def serve_zoomed_face(filename):
    return send_from_directory(MATCHED_FACES_ZOOM_DIR, filename)

@app.route('/faces/<filename>')
def send_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

FACES_DIR = "faces"

@app.route('/rename_image', methods=['POST'])
def rename_image():
    data = request.json
    old_filename = data['oldFilename']
    new_filename = data['newFilename']

    old_path = os.path.join(FACES_DIR, old_filename)
    new_path = os.path.join(FACES_DIR, new_filename)

    if not os.path.exists(old_path):
        return jsonify({"success": False, "error": "Old file does not exist"})

    try:
        os.rename(old_path, new_path)

        old_name = os.path.splitext(old_filename)[0]
        new_name = os.path.splitext(new_filename)[0]

        #  Update known_names list
        if old_name in known_names:
            idx = known_names.index(old_name)
            known_names[idx] = new_name

            #  Update META file
            if os.path.exists(META_FILE):
                with open(META_FILE, 'r') as f:
                    meta = json.load(f)
            else:
                meta = {}

            if old_filename in meta:
                meta[new_filename] = meta.pop(old_filename)

            #  Save updates to cache
            with open(NAMES_FILE, 'w') as f:
                json.dump(known_names, f)

            with open(META_FILE, 'w') as f:
                json.dump(meta, f)

        #  Update matched_faces list
        for face in matched_faces:
            if face['name'] == old_name:
                face['name'] = new_name
                face['img_url1'] = f"/matched_faces/{new_filename}"
                face['img_url2'] = f"/matched_faces_zoom/{new_filename}"
                face['img_url3'] = f"/faces/{new_filename}"

                # Rename corresponding matched face image files
                old_matched = os.path.join(MATCHED_FACES_DIR, old_filename)
                new_matched = os.path.join(MATCHED_FACES_DIR, new_filename)
                if os.path.exists(old_matched):
                    os.rename(old_matched, new_matched)

                old_zoom = os.path.join(MATCHED_FACES_ZOOM_DIR, old_filename)
                new_zoom = os.path.join(MATCHED_FACES_ZOOM_DIR, new_filename)
                if os.path.exists(old_zoom):
                    os.rename(old_zoom, new_zoom)

                break  # Name is unique; stop after update

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/delete_image', methods=['POST'])
def delete_image():
    data = request.json
    file_path = os.path.join(FACES_DIR, data['filename'])

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"success": True})
    return jsonify({"success": False})

if __name__ == '__main__':
    load_known_faces()
    # Start face recognition worker thread
    face_thread = threading.Thread(target=face_recognition_worker, daemon=True)
    face_thread.start()

    app.run(debug=True)
