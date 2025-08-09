import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load model MobileNetV2
model = MobileNetV2(weights='imagenet')

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
IMG_SIZE = 224

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediksi
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=1)[0]
    label = decoded_preds[0][1]
    confidence = decoded_preds[0][2]

    # Tampilkan hasil
    display_text = f"{label}: {confidence*100:.2f}%"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Deteksi Objek', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
