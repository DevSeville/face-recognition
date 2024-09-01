import os
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cv2
code_dir = os.path.abspath('.')
train_image_dir = code_dir+'/train-images'
test_image_dir = code_dir+'/test-images'

for i in tqdm(range(len(os.listdir(train_image_dir))), desc="locating train images"):
    time.sleep(0.1)
for b in tqdm(range(len(os.listdir(test_image_dir))), desc="locating test images"):
    time.sleep(0.1)

def load_images_from_directory(directory):
    image_files = [
        os.path.join(directory, file) for file in os.listdir(directory)
        if file.endswith((".jpg", ".jpeg", ".png", ".svg", ".jfif"))
    ]
    
    for filename in image_files:
        img = face_recognition.load_image_file(filename)
        print(f"Loaded image: {filename}, dtype: {img.dtype}, shape: {img.shape}")

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print(f"Converted image: {filename}, dtype: {rgb_img.dtype}, shape: {rgb_img.shape}")
        
        try:
            face_locations = face_recognition.face_locations(rgb_img)
            print(f"Found faces in {filename}: {face_locations}")
            img_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            print(f"Found {len(img_encodings)} faces in {filename}.")
            for img_encoding in img_encodings:
                yield img_encoding, os.path.basename(filename), rgb_img 
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
train_embeddings = []
train_labels = []
train_images = [] 

for encoding, label, img in tqdm(load_images_from_directory(train_image_dir), desc="Loading training images"):
    if img.ndim == 3 and img.shape[2] in [3, 4]:  
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_encodings = face_recognition.face_encodings(img_rgb)
        
        if img_encodings:
            train_embeddings.append(img_encodings[0])
            train_labels.append(label)
            train_images.append(img)
    else:
        print(f"Image format not supported for {label}, skipping...")

print(f"Loaded {len(train_images)} images with {len(train_embeddings)} embeddings.")

test_embeddings = []
test_images = []

for encoding, label, img in tqdm(load_images_from_directory(test_image_dir), desc="Loading test images"):
    test_embeddings.append(encoding)
    test_images.append(img)

print(f"Loaded {len(test_images)} test images with {len(test_embeddings)} embeddings.")

recognized_faces = []
matched_training_images = []

for test_embedding in tqdm(test_embeddings, desc="Recognizing faces in test images"):
    distances = face_recognition.face_distance(train_embeddings, test_embedding)

    if distances.size > 0:
        best_match_index = np.argmin(distances)
        recognized_faces.append(train_labels[best_match_index])
        matched_training_images.append(train_images[best_match_index])
    else:
        recognized_faces.append("Unknown")
        matched_training_images.append(None)

print(f"Recognized {len(recognized_faces)} faces.")

print("Recognized faces:")
for i, face in enumerate(recognized_faces):
    print(f"Test image {i}: Recognized face - {face}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)

    img_rgb = cv2.cvtColor(test_images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb) 
    plt.title(f"Test Image {i}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if matched_training_images[i] is not None:
        matched_img_rgb = cv2.cvtColor(matched_training_images[i], cv2.COLOR_BGR2RGB)
        plt.imshow(matched_img_rgb)
        plt.title(f"Matched Face: {face}")
    else:
        plt.title("No Match Found")
    
    plt.axis('off')
    plt.show()