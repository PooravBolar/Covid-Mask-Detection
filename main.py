import cv2
import numpy as np
import tensorflow as tf
import keras

# Load the pre-trained model
model = keras.models.load_model('mask_detection_model.h5')

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the frame
def preprocess_frame(frame, x, y, w, h):
    # Crop the face region
    face_roi = frame[y:y+h, x:x+w]
    # Resize to match model input size
    resized_face = cv2.resize(face_roi, (224, 224))

    return resized_face

# Access the camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Preprocess the face region
        processed_face = preprocess_frame(frame, x, y, w, h)
        
        # Make predictions
        predictions = model.predict(np.expand_dims(processed_face, axis=0))
        
        # Draw bounding box and label based on predictions
        if predictions[0][0] > 0.5:  # Example threshold
            label = "Mask"
            color = (0, 255, 0)  # Green for mask
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red for no mask
        
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Put label
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the result
    cv2.imshow('Mask Detection', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
