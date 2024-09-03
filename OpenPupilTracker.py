import cv2
import dlib
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys
import os
from datetime import datetime

# Cargar el detector de caras y el predictor de puntos faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Función para obtener la región de interés del ojo
def get_eye_region(shape, eye_points):
    points = [(shape.part(point).x, shape.part(point).y) for point in eye_points]
    return points

# Función optimizada para detectar la posición de la pupila
def detect_pupil(eye_image):
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

# Función para procesar un ojo (para usar con multithreading)
def process_eye(eye_image):
    return detect_pupil(eye_image)

# Clase simple para el filtro de Kalman
class SimpleKalmanFilter:
    def __init__(self, measurement_uncertainty, estimation_uncertainty, process_uncertainty):
        self.measurement_uncertainty = measurement_uncertainty
        self.estimation_uncertainty = estimation_uncertainty
        self.process_uncertainty = process_uncertainty
        self.estimate = 0
        self.estimation_uncertainty = estimation_uncertainty

    def update(self, measurement):
        kalman_gain = self.estimation_uncertainty / (self.estimation_uncertainty + self.measurement_uncertainty)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.estimation_uncertainty = (1 - kalman_gain) * self.estimation_uncertainty + abs(self.estimate - measurement) * self.process_uncertainty
        return self.estimate

# Función para guardar la imagen de seguimiento de mirada
def save_gaze_tracking_image(image, prefix="gaze_tracking"):
    if not os.path.exists("gaze_tracking_images"):
        os.makedirs("gaze_tracking_images")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"gaze_tracking_images/{prefix}-{timestamp}.png"
    cv2.imwrite(filename, image)
    print(f"Imagen de seguimiento de mirada guardada como {filename}")

# Pedir la IP y el puerto al usuario
ip = input("Introduce la dirección IP de la cámara web: ")
port = input("Introduce el puerto de la cámara web: ")
camera_url = f"http://{ip}:{port}/video"

# Inicializar la captura de video
cap = cv2.VideoCapture(camera_url)

# Verificar si la conexión fue exitosa
if not cap.isOpened():
    print(f"Error: No se pudo conectar a la cámara web en {camera_url}")
    sys.exit(1)

# Crear una ventana separada para dibujar el seguimiento de la mirada
drawing_window = np.zeros((500, 500, 3), dtype=np.uint8)
window_center = (drawing_window.shape[1] // 2, drawing_window.shape[0] // 2)
previous_position = window_center

# Factor de escala para ampliar el movimiento
scale_factor = 35

# Buffer circular para suavizar el movimiento
buffer_size = 5
position_buffer = deque(maxlen=buffer_size)

# Configuración para procesar menos frames
frame_skip = 2
frame_count = 0

# Inicializar filtros de Kalman simples para x e y
kf_x = SimpleKalmanFilter(2, 2, 0.01)
kf_y = SimpleKalmanFilter(2, 2, 0.01)

# Contador para los primeros puntos de mirada
initial_points_count = 0

print("Presiona 'q' para salir del programa y guardar la imagen de seguimiento de mirada.")

with ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error al leer el frame de la cámara")
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)

                left_eye_region = get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])
                right_eye_region = get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])
                
                left_eye = frame[min([p[1] for p in left_eye_region]):max([p[1] for p in left_eye_region]), 
                                 min([p[0] for p in left_eye_region]):max([p[0] for p in left_eye_region])]
                right_eye = frame[min([p[1] for p in right_eye_region]):max([p[1] for p in right_eye_region]), 
                                  min([p[0] for p in right_eye_region]):max([p[0] for p in right_eye_region])]

                left_future = executor.submit(process_eye, left_eye)
                right_future = executor.submit(process_eye, right_eye)
                
                left_pupil = left_future.result()
                right_pupil = right_future.result()

                if left_pupil and right_pupil:
                    avg_pupil_position = ((left_pupil[0] + right_pupil[0]) // 2, (left_pupil[1] + right_pupil[1]) // 2)
                    
                    # Calcular el desplazamiento desde el centro del ojo
                    eye_center = (left_eye.shape[1] // 2, left_eye.shape[0] // 2)
                    displacement = (avg_pupil_position[0] - eye_center[0], avg_pupil_position[1] - eye_center[1])
                    
                    # Aplicar el factor de escala y añadir al centro de la ventana
                    drawing_point = (
                        int(window_center[0] + displacement[0] * scale_factor),
                        int(window_center[1] + displacement[1] * scale_factor)
                    )
                    
                    # Asegurar que el punto está dentro de los límites de la ventana
                    drawing_point = (
                        max(0, min(drawing_point[0], drawing_window.shape[1] - 1)),
                        max(0, min(drawing_point[1], drawing_window.shape[0] - 1))
                    )
                    
                    # Aplicar suavizado con buffer circular
                    position_buffer.append(drawing_point)
                    smoothed_point = tuple(map(int, np.mean(position_buffer, axis=0)))
                    
                    # Aplicar filtro de Kalman simple
                    filtered_x = kf_x.update(smoothed_point[0])
                    filtered_y = kf_y.update(smoothed_point[1])
                    filtered_point = (int(filtered_x), int(filtered_y))
                    
                    # No pintar los primeros 10 puntos, fallo encontrado de precisión en estos primeros puntos
                    if initial_points_count >= 10:
                        cv2.line(drawing_window, previous_position, filtered_point, (0, 255, 0), 2)
                    else:
                        initial_points_count += 1
                    
                    previous_position = filtered_point
            
            cv2.imshow('Frame', frame)
            cv2.imshow('Gaze Tracking', drawing_window)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                save_gaze_tracking_image(drawing_window, prefix="OpenPupilTracker")
                break

        except Exception as e:
            print(f"Error: {str(e)}")
            print("Se ha perdido la conexión con el dispositivo.")
            save_gaze_tracking_image(drawing_window, prefix="OpenPupilTracker")
            print("La imagen de seguimiento de mirada ha sido guardada.")
            break

cap.release()
cv2.destroyAllWindows()