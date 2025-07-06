
# AI-Based Two-Way Communication System for the Deaf

_Real-time Sign Language & Speech Interface with Translation and Audio Output_

## 🧠 Overview
This project aims to bridge the communication gap between hearing-impaired and hearing individuals using artificial intelligence. It enables real-time two-way communication using:

- ✋ **Sign Language Recognition** (Gesture to Speech)
- 🗣️ **Speech Recognition** (Speech to Sign/Text)
- 🌐 **Translation** (English to Tamil)
- 🔊 **Text-to-Speech** (Tamil audio output)

## 🧰 Tech Stack

| Component             | Tool/Library              |
|-----------------------|---------------------------|
| Web Framework         | Flask                     |
| Hand Tracking         | MediaPipe                 |
| Gesture Recognition   | CNN + LSTM (Keras/TensorFlow) |
| Speech Recognition    | SpeechRecognition (Google Web Speech API) |
| Translation           | Google Translate API      |
| Text-to-Speech (TTS)  | gTTS / Tacotron2          |
| Dataset               | WLASL (Word-Level ASL)    |

## 📂 Dataset: WLASL
We used a curated subset of the [WLASL dataset](https://github.com/dxli94/WLASL), which includes video samples for over 2,000 words in American Sign Language.

✅ Preprocessing Steps:
- Converted videos into image sequences (128×128, grayscale)
- Extracted landmarks using MediaPipe
- Labeled and mapped to English → Tamil using Google Translate

## 🔁 System Flow

### 🔹 Speech-to-Sign (Normal user → Deaf)
1. 🎤 User speaks (English)
2. 📝 Speech converted to text (Google Speech API)
3. 🌐 Text translated to Tamil (Google Translate API)
4. 🔊 Tamil speech output generated (gTTS/Tacotron2)

### 🔹 Sign-to-Speech (Deaf → Normal user)
1. 🎥 Camera captures hand gestures
2. ✋ MediaPipe extracts hand landmarks
3. 🧠 CNN-LSTM model classifies gesture
4. 🌐 Word translated to Tamil
5. 🔊 Tamil speech output

## 🚀 Getting Started

### 🔧 Prerequisites
```bash
pip install flask mediapipe opencv-python tensorflow keras gtts SpeechRecognition googletrans==4.0.0-rc1
```

### ▶️ Run the App
```bash
python app.py
```

- Open browser: `http://127.0.0.1:5000`
- Use either interface: **Speech input** or **Gesture recognition**

## 📈 Model Performance

| Task                 | Accuracy / Score      |
|----------------------|-----------------------|
| Gesture Recognition  | ~87% (CNN-LSTM)       |
| Speech-to-Text       | ~94% (Google Speech API) |
| Translation          | ~90% BLEU (EN→TA)     |
| Text-to-Speech       | ~4.3/5 MOS (gTTS/Tacotron2) |

## 📚 References
- [1] Reddy et al., *AIDE 2025* - Real-time Sign Recognition using MediaPipe and Random Forest  
- [2] Rajkumar et al., *ICEARS 2023* - AI-Based Two-Way Sign Language System  
- [3] WLASL Dataset: https://github.com/dxli94/WLASL  
- [4] MediaPipe by Google: https://mediapipe.dev  
- [5] TensorFlow/Keras: https://www.tensorflow.org

## 👥 Authors
- **Priyadharshini G** – Developer & ML Engineer  
