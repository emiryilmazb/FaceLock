import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
from time import sleep

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
RED_LED_PIN = 24
GREEN_LED_PIN = 23
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)

# Function to turn on the red LED
def red_led_on():
    GPIO.output(RED_LED_PIN, GPIO.HIGH)
    GPIO.output(GREEN_LED_PIN, GPIO.LOW)

# Function to turn on the green LED
def green_led_on():
    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
    GPIO.output(RED_LED_PIN, GPIO.LOW)

# Function to load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}
    label_count = 0
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            label = filename.split('.')[0]  # Assuming filenames are in the format 'username.jpg'
            if label not in label_dict:
                label_dict[label] = label_count
                label_count += 1
            labels.append(label_dict[label])
    return images, labels, label_dict

# Load known user images and labels
folder_path = 'users'
images, labels, label_dict = load_images_from_folder(folder_path)
images = [cv2.resize(img, (200, 200)) for img in images]

# Create and train face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(images, np.array(labels))

# Initialize camera
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    while True:
        red_led_on()
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))

            label, confidence = face_recognizer.predict(face_resized)
            print(f"Label: {label}, Confidence: {confidence}")

            if confidence < 50:  # Confidence threshold, adjust as needed
                user_name = [name for name, value in label_dict.items() if value == label][0]
                print(f"The door is opening for {user_name}")
                green_led_on()
                sleep(5)  # Keep green LED on for 5 seconds
            else:
                red_led_on()

        sleep(1)  # Check every second
finally:
    camera.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
