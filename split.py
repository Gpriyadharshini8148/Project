import os
import shutil
import random

SOURCE_DIR = 'D:\\PROJECT\\TWO WAY\\datasets\\images'
TRAIN_DIR = 'D:\\PROJECT\\TWO WAY\\train_dataset'
TEST_DIR = 'D:\\PROJECT\\TWO WAY\\test_dataset'
SPLIT_RATIO = 0.2

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        print(f"Skipping non-folder: {class_path}")
        continue

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Class '{class_name}' has {len(image_files)} images")

    random.shuffle(image_files)
    split_point = int(len(image_files) * SPLIT_RATIO)
    test_images = image_files[:split_point]
    train_images = image_files[split_point:]

    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TRAIN_DIR, class_name, img)
        try:
            shutil.copyfile(src, dst)
            print(f"✅ Copied train: {dst}")
        except Exception as e:
            print(f"❌ Could not copy {src}: {e}")

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TEST_DIR, class_name, img)
        try:
            shutil.copyfile(src, dst)
            print(f"✅ Copied test: {dst}")
        except Exception as e:
            print(f"❌ Could not copy {src}: {e}")

print("✅ Image dataset split completed.")
