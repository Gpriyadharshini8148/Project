from flask import Flask, render_template, Response, request, jsonify, url_for, redirect
import cv2
import threading
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import time
import uuid
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Modules
recognizer = sr.Recognizer()
translator = Translator()

# MediaPipe Hand Model (support both hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Model settings
MODEL_PATH = "model1.h5"
LABEL_MAP_PATH = "label_map.txt"
IMG_SIZE = 128

# Globals
deaf_to_normal_text = recognized_english_text = normal_to_deaf_text = ""
deaf_audio_filename = ""
last_deaf_timestamp = 0
gesture_text = ""
input_mode = "speech"
model = None
label_map = {}

# Create audio folder if not exists
os.makedirs("static/audio", exist_ok=True)

# Load CNN Model and Label Map
def load_pretrained_model():
    global model, label_map
    model = load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = eval(f.read())
    print("Loaded model with labels:", len(label_map))

# Recognize gestures from both hands using CNN + MediaPipe
def recognize_gesture(frame):
    global gesture_text
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    gesture_texts = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                preds = model.predict(resized)
                idx = np.argmax(preds)
                label = label_map.get(idx, "")
                gesture_texts.append(label)

    gesture_text = " + ".join(gesture_texts)
    return frame

# Live video feed for sign recognition
def generate_frames():
    global deaf_to_normal_text, recognized_english_text, deaf_audio_filename, last_deaf_timestamp

    while True:
        success, frame = camera.read()
        if not success:
            break

        if input_mode == "sign":
            frame = recognize_gesture(frame)
            if gesture_text:
                recognized_english_text = gesture_text
                translated = translator.translate(gesture_text, src='en', dest='ta')
                deaf_to_normal_text = translated.text
                last_deaf_timestamp = int(time.time())

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Speech-to-text and translation background thread
def speech_thread():
    global deaf_to_normal_text, deaf_audio_filename, last_deaf_timestamp, recognized_english_text
    while True:
        if input_mode == "speech":
            try:
                with sr.Microphone() as source:
                    print("Listening...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    text = recognizer.recognize_google(audio, language="en-IN")
                    recognized_english_text = text
                    translated = translator.translate(text, src='en', dest='ta')
                    deaf_to_normal_text = translated.text
                    last_deaf_timestamp = int(time.time())

                    filename = f"{uuid.uuid4().hex}.mp3"
                    tts = gTTS(text=deaf_to_normal_text, lang='ta')
                    tts.save(os.path.join("static/audio", filename))
                    deaf_audio_filename = filename

            except Exception as e:
                print("Speech error:", e)
        time.sleep(1)

# Start background speech thread
threading.Thread(target=speech_thread, daemon=True).start()

# ------------------ Flask Routes ------------------ #
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/deaf')
def deaf():
    return render_template("deaf.html")

@app.route('/normal')
def normal():
    return render_template("normal.html")

@app.route('/accuracy')
def accuracy():
    return render_template("accuracy.html", time=int(time.time()))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_deaf_to_normal')
def get_deaf_to_normal():
    return jsonify({
        'recognized_text': recognized_english_text,
        'translated_text': deaf_to_normal_text,
        'audio_url': url_for('static', filename=f'audio/{deaf_audio_filename}') if deaf_audio_filename else '',
        'timestamp': last_deaf_timestamp
    })

@app.route('/normal_reply', methods=['POST'])
def normal_reply():
    global normal_to_deaf_text
    data = request.get_json()
    tamil_text = data.get("text", "")
    translated = translator.translate(tamil_text, src='ta', dest='en')
    normal_to_deaf_text = translated.text
    return jsonify({"translated_text": normal_to_deaf_text})

@app.route('/get_normal_to_deaf')
def get_normal_to_deaf():
    return jsonify({'translated_text': normal_to_deaf_text})

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global input_mode
    if mode in ["speech", "sign"]:
        input_mode = mode
    return redirect(url_for('deaf'))

# ------------------ Run App ------------------ #
if __name__ == "__main__":
    load_pretrained_model()
    app.run(debug=True)
