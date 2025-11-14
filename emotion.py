import cv2
from deepface import DeepFace

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("⚠️ No face detected in the image.")
        return

    for (x, y, w, h) in faces:
        face_roi = img[y:y + h, x:x + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save output image only once
    save_path = "output_emotion.jpg"
    cv2.imwrite(save_path, img)

    import os
    full_path = os.path.abspath(save_path)
    print(f"✅ Image saved at: {full_path}")


    # Show image only once
    cv2.imshow("Emotion Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def analyze_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Real-time Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = input("Enter mode ('camera' or 'image'): ").strip().lower()

    if mode == "camera":
        analyze_webcam()
    elif mode == "image":
        image_path = input("Enter image filename or path (e.g. test.jpg): ").strip()
        analyze_image(image_path)
    else:
        print("❌ Invalid mode! Please enter 'camera' or 'image'.")
