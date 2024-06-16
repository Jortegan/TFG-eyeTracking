import cv2
import dlib
import numpy as np
from math import hypot

# Cargar el detector de rostros de Dlib y el predictor de forma
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Asegúrate de que este archivo esté en el mismo directorio

threshold_value = 70  # Valor inicial del umbral
canvas_size = 500  # Tamaño del lienzo para la visualización de la dirección de la mirada

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
    # Obtener las coordenadas de los puntos de los ojos
    x_coords = [landmarks.part(point).x for point in eye_points]
    y_coords = [landmarks.part(point).y for point in eye_points]

    # Calcular los límites del área del ojo con un pequeño margen
    x_min = min(x_coords) - 5
    x_max = max(x_coords) + 5
    y_min = min(y_coords) - 5
    y_max = max(y_coords) + 5

    # Extraer la región del ojo del cuadro original
    eye_region = frame[y_min:y_max, x_min:x_max]
    return eye_region, (x_min, y_min, x_max, y_max)

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_eye_top_down_looking(eye_points, landmarks):
    center_top = midpoint(landmarks.part(eye_points[0]), landmarks.part(eye_points[1]))
    center_bottom = midpoint(landmarks.part(eye_points[2]), landmarks.part(eye_points[3]))
    return hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

def detect_eye_direction(frame, eye_points, facial_landmarks, threshold_value):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    normal_eye = frame[min_y: max_y, min_x: max_x]

    gray_eye = cv2.cvtColor(normal_eye, cv2.COLOR_BGR2GRAY)

    # Filtrado de imagen
    gray_eye = cv2.GaussianBlur(gray_eye, (11, 11), 0)
    gray_eye = cv2.bilateralFilter(gray_eye, 11, 11, 11)

    if threshold_value % 2 == 0:
        threshold_value += 1

    threshold_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold_value, 10)

    kernel = np.ones((3, 3), np.uint8)
    threshold_eye = cv2.erode(threshold_eye, kernel, iterations=1)

    kernel = np.ones((7, 7), np.uint8)
    threshold_eye = cv2.morphologyEx(threshold_eye, cv2.MORPH_CLOSE, kernel)

    th, threshold_eye = cv2.threshold(threshold_eye, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, hierarchy = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) != 0:
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)

        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            center = (int(cX), int(cY))
            cv2.circle(normal_eye, center, 7, (0, 255, 0), 2)
        except:
            cX = -1

        return cX, threshold_eye, normal_eye

    else:
        return -1, threshold_eye, normal_eye

def draw_gaze_direction(canvas, hor_dir, ver_dir, canvas_size):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype="uint8")
    center_x = int(canvas_size / 2)
    center_y = int(canvas_size / 2)
    
    # Mapear las coordenadas de hor_dir y ver_dir al tamaño del canvas
    gaze_x = int(center_x + (hor_dir - center_x) * 2)  # Multiplicador para ajustar sensibilidad
    gaze_y = int(center_y + (ver_dir - center_y) * 2)
    
    cv2.circle(canvas, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
    return canvas

def main():
    global threshold_value
    cap = cv2.VideoCapture("http://172.20.10.7:4747/video")  # Para video en tiempo real desde la cámara. Para archivo de video, reemplace con la ruta del archivo.

    show_landmarks = True
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype="uint8")

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
            
            # Dibujar los puntos de referencia en el cuadro si está habilitado
            if show_landmarks:
                draw_landmarks(frame, landmarks)

            # Extraer y mostrar regiones de los ojos
            left_eye_region, left_eye_coords = extract_eye_region(frame, landmarks, range(36, 42))
            right_eye_region, right_eye_coords = extract_eye_region(frame, landmarks, range(42, 48))

            # Redimensionar las regiones de los ojos para una mejor visualización
            left_eye_resized = cv2.resize(left_eye_region, (left_eye_region.shape[1]*4, left_eye_region.shape[0]*4))
            right_eye_resized = cv2.resize(right_eye_region, (right_eye_region.shape[1]*4, right_eye_region.shape[0]*4))

            # Detectar la dirección de la mirada y mostrar el resultado
            lCx, bwLeftEye, leftEye = detect_eye_direction(frame, range(36, 42), landmarks, threshold_value)
            rCx, bwRightEye, rightEye = detect_eye_direction(frame, range(42, 48), landmarks, threshold_value)

            hor_dir = (lCx + rCx) / 2

            lCy = get_eye_top_down_looking([37, 38, 41, 40], landmarks)
            rCy = get_eye_top_down_looking([43, 44, 47, 46], landmarks)

            ver_dir = (lCy + rCy) / 2

            # Actualizar el lienzo con la dirección de la mirada
            canvas = draw_gaze_direction(canvas, hor_dir, ver_dir, canvas_size)

            # Redimensionar las imágenes procesadas para mejor visualización
            left_eye_threshold_resized = cv2.resize(bwLeftEye, (bwLeftEye.shape[1]*4, bwLeftEye.shape[0]*4))
            right_eye_threshold_resized = cv2.resize(bwRightEye, (bwRightEye.shape[1]*4, bwRightEye.shape[0]*4))

            # Mostrar los resultados
            cv2.imshow('Thresholded Left Eye', left_eye_threshold_resized)
            cv2.imshow('Thresholded Right Eye', right_eye_threshold_resized)
            cv2.imshow('Left Eye', leftEye)
            cv2.imshow('Right Eye', rightEye)

        # Mostrar el lienzo de dirección de la mirada
        cv2.imshow('Gaze Direction', canvas)

        # Mostrar el marco original con puntos de referencia
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif key == ord('+'):
            threshold_value += 1
        elif key == ord('-'):
            threshold_value -= 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
