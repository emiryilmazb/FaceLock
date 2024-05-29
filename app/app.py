import cv2
import face_recognition
import os
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2

# GPIO setup
RED_LED_PIN = 24
GREEN_LED_PIN = 23
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)

# Function to initialize the Raspberry Pi camera module using picamera2
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.start()
        return picam2
    except Exception as e:
        print(f"Error initializing camera: {e}")
        GPIO.cleanup()
        exit(1)

# Function to load and encode user images
def load_user_images(users_path="users"):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(users_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(users_path, filename)
            user_image = face_recognition.load_image_file(image_path)
            user_face_encodings = face_recognition.face_encodings(user_image)
            
            if user_face_encodings:
                user_face_encoding = user_face_encodings[0]
                known_face_encodings.append(user_face_encoding)
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"No faces found in image {filename}")
                
    return known_face_encodings, known_face_names

# Function to control LEDs
def set_leds(red_on, green_on):
    GPIO.output(RED_LED_PIN, red_on)
    GPIO.output(GREEN_LED_PIN, green_on)

def main():
    # Initialize camera
    picam2 = initialize_camera()

    # Load user images
    known_face_encodings, known_face_names = load_user_images()

    # Initialize LEDs
    set_leds(red_on=True, green_on=False)

    while True:
        # Capture image from the camera
        frame = picam2.capture_array()

        # Ensure the image is in RGB format
        if frame.shape[2] == 4:  # If the image has an alpha channel (RGBA), convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:  # Otherwise, convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame for faster processing
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

        # Check if any authorized faces are recognized
        if "Unknown" in face_names or not face_names:
            set_leds(red_on=True, green_on=False)
        else:
            set_leds(red_on=False, green_on=True)
            for name in face_names:
                if name != "Unknown":
                    print(f"The door is opened for {name}")
                    break
            time.sleep(5)  # Keep green LED on for 5 seconds
            set_leds(red_on=True, green_on=False)

        # Draw face boxes and display names on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the camera and cleanup GPIO
    picam2.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
