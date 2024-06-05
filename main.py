import face_recognition
import cv2
import numpy as np

# Load the known image and encode it
try:
    known_image = face_recognition.load_image_file("leclerc.jpeg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]
    print("Known face encoding loaded successfully.")
except Exception as e:
    print(f"Error loading known image: {e}")
    exit(1)

# Create arrays of known face encodings and their names
known_face_encodings = [
    known_face_encoding,
]
known_face_names = [
    "Charles",
]

# Load the input image to test
try:
    input_image = face_recognition.load_image_file("input_image.jpeg")
    input_image_rgb = input_image[:, :, ::-1]  # Convert BGR (OpenCV) to RGB
    print("Input image loaded successfully.")
except Exception as e:
    print(f"Error loading input image: {e}")
    exit(1)

# Debug step: Display the input image to ensure it is loaded correctly
try:
    input_image_bgr = input_image[:, :, ::-1]  # Convert RGB to BGR for OpenCV display
    cv2.imshow('Loaded Input Image', input_image_bgr)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    print("Input image displayed successfully.")
except Exception as e:
    print(f"Error displaying input image: {e}")
    exit(1)

# Find all the faces and face encodings in the input image
try:
    face_locations = face_recognition.face_locations(input_image_rgb, model="cnn")
    face_encodings = face_recognition.face_encodings(input_image_rgb, face_locations)
    print(f"Found {len(face_locations)} face(s) in the input image.")
except Exception as e:
    print(f"Error processing input image: {e}")
    exit(1)

# If no faces are detected, try with a different face detection model
if len(face_locations) == 0:
    print("No faces found using the default face detection model. Trying with the HOG model.")
    try:
        face_locations = face_recognition.face_locations(input_image_rgb, model="hog")
        face_encodings = face_recognition.face_encodings(input_image_rgb, face_locations)
        print(f"Found {len(face_locations)} face(s) in the input image using the HOG model.")
    except Exception as e:
        print(f"Error processing input image using the HOG model: {e}")
        exit(1)

# If still no faces are detected, exit
if len(face_locations) == 0:
    print("No faces found in the input image.")
    exit(1)

# Display the face locations found in the input image
for i, (top, right, bottom, left) in enumerate(face_locations):
    print(f"Face {i+1} found at Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")

face_names = []
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)

# Display the results
# Display the results
try:
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Convert coordinates to match the original image size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(input_image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
        print(f"Rectangle drawn at Left: {left}, Top: {top}, Right: {right}, Bottom: {bottom} with name: {name}")

        # Draw a label with a name below the face
        cv2.rectangle(input_image_bgr, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(input_image_bgr, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print(f"Label {name} drawn below face")

    # Display the resulting image
    cv2.imshow('Result', input_image_bgr)
    print("Displaying the result. Press 'q' to exit.")

    # Hit 'q' on the keyboard to quit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error displaying result image: {e}")
finally:
    # Cleanup
    cv2.destroyAllWindows()
    print("Cleaned up and exited successfully.")


