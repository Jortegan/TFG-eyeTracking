import cv2
import dlib

# Cargar el detector de rostros de Dlib y el predictor de forma
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Asegúrate de que este archivo esté en el mismo directorio

def draw_landmarks(frame, landmarks):
    # Puntos faciales
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        if i in range(36, 48):  # Puntos de los ojos
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Verde
        else:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Rojo

def extract_eye_region(frame, landmarks, eye_points):
    x_coords = [landmarks.part(point).x for point in eye_points]
    y_coords = [landmarks.part(point).y for point in eye_points]

    x_min = min(x_coords) - 5
    x_max = max(x_coords) + 5
    y_min = min(y_coords) - 5
    y_max = max(y_coords) + 5

    eye_region = frame[y_min:y_max, x_min:x_max]
    return eye_region

def main():
    cap = cv2.VideoCapture("http://172.20.10.7:4747/video")  # Para video en tiempo real desde la cámara. Para archivo de video, reemplace con la ruta del archivo.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Invertir la imagen horizontalmente
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)

            # Extraer y mostrar regiones de los ojos
            left_eye_region = extract_eye_region(frame, landmarks, range(36, 42))
            right_eye_region = extract_eye_region(frame, landmarks, range(42, 48))

            # Redimensionar las regiones de los ojos para una mejor visualización
            left_eye_resized = cv2.resize(left_eye_region, (left_eye_region.shape[1]*4, left_eye_region.shape[0]*4))
            right_eye_resized = cv2.resize(right_eye_region, (right_eye_region.shape[1]*4, right_eye_region.shape[0]*4))

            cv2.imshow('Ojo Izquierdo', left_eye_resized)
            cv2.imshow('Ojo Derecho', right_eye_resized)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
