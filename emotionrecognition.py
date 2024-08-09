import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load the pre-trained model for emotion recognition
model = load_model('emotion_detection_model.h5')

# Define the list of emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video capture
cap = cv2.VideoCapture(0)  # Change to the video file path if using a video

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face region to match the input size of the model
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize the pixel values to be in the range [0, 1]
        face_roi = face_roi / 255.0

        # Reshape the face to match the input shape expected by the model
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Perform emotion prediction
        emotion_pred = model.predict(face_roi)
        emotion_label = emotion_labels[np.argmax(emotion_pred)]

        # Draw the bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
