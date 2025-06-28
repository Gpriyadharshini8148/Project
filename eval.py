import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128
MODEL_PATH = "model1.h5"
LABEL_MAP_PATH = "label_map.txt"
TEST_DIR = "D:\\PROJECT\\TWO WAY\\test_dataset"

# Load model
model = load_model(MODEL_PATH)

# Load label map
with open(LABEL_MAP_PATH, "r") as f:
    label_map = eval(f.read())
inv_label_map = {v: k for k, v in label_map.items()}

total = 0
correct = 0

for class_name in os.listdir(TEST_DIR):
    class_path = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"🔍 Evaluating class: {class_name}")
    for filename in os.listdir(class_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(class_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None or img.size == 0:
            print(f"⚠️ Skipped unreadable image: {img_path}")
            continue

        try:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

            pred = model.predict(img, verbose=0)
            pred_class = np.argmax(pred)
            pred_label = str(label_map[pred_class])

            print(f"🔎 Predicted: {pred_label} | Actual: {class_name}")

            total += 1
            if pred_label.lower() == class_name.lower():
                correct += 1

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

# Accuracy
if total > 0:
    accuracy = correct / total * 100
    print("\n🎯 Evaluation Results:")
    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"✅ Correct Predictions: {correct}")
    print(f"📊 Total Samples: {total}")
else:
    print("❌ No valid test images found.")
