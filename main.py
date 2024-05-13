import cv2
import face_recognition
import numpy as np
import random
import os
from PIL import Image

# Load known faces from the "us" folder
known_face_encodings = []
known_face_names = []

# Path to the folder containing known face images
folder_path = "us"

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]  # Assuming there's only one face in each image
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use the filename as the name

# Randomly choose the number of blinks required
num_blinks_required = random.randint(3, 5)

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Multi-view face recognition
        if name != "Unknown":
            # Ask the user to look left and right
            cv2.putText(frame, "Look left and right", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Randomly blinking simulation
            if random.random() < 0.5:
                num_blinks_required -= 1

            if num_blinks_required <= 0:
                # Ask the user to perform facial expression
                cv2.putText(frame, "Perform facial expression", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Simulate facial expression detection
                expression_detected = random.choice(["smile", "eyebrow_move"])
                if expression_detected == "smile" or expression_detected == "eyebrow_move":
                    # Authentication successful
                    cv2.putText(frame, "Authentication successful!", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
