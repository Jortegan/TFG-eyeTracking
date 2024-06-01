# import cv2
# import numpy as np
# import os
# from datetime import datetime
# import dlib

# # Inicializar el detector de caras y el predictor de puntos de referencia faciales
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# capture = cv2.VideoCapture("http://192.168.1.51:4747/video")

# if not capture.isOpened():
#     print("Error al abrir el video.")
# else:
#     # Variables que usaremos
#     rotation_angle = 0  # Declaramos la variable de la rotacion, por defecto, es como salga la imagen
#     recording = False   # Modo de grabado, por defecto está apagado
#     out = None          # Salida del modo de grabacion, por defecto desactivado

#     while True:
#         ret, frame = capture.read()

#         # PASO 1: Comprobamos que tenemos acceso a la camara IP
#         if not ret:
#             print("No se pudo leer el cuadro del video.")
#             break

#         # PASO 2: Rotamos la imagen para que se adapte a nuestra pantalla (pulsando tecla R)
#         if cv2.waitKey(1) == ord("r"):
#             rotation_angle += 90
#             if rotation_angle == 360:
#                 rotation_angle = 0

#         # Obtener dimensiones de la imagen
#         height, width = frame.shape[:2]
        
#         # Calcular el centro de la imagen
#         center = (width / 2, height / 2)
        
#         # Definir la matriz de rotación
#         rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        
#         # Aplicar la transformación de rotación
#         rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

#         # Convertir la imagen a escala de grises para la detección facial
#         gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

#         # Detectar caras en la imagen
#         faces = detector(gray)

#         # Iterar sobre las caras detectadas
#         for face in faces:
#             # Obtener los puntos de referencia faciales
#             landmarks = predictor(gray, face)

#             # Los puntos 36-41 son para el ojo izquierdo y los puntos 42-47 son para el ojo derecho
#             left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
#             right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

#             # Calcular el centro de los ojos izquierdo y derecho
#             left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
#             right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

#             # Dibujar un círculo en el centro de los ojos
#             cv2.circle(rotated_frame, tuple(left_eye_center), 2, (0, 255, 0), -1)
#             cv2.circle(rotated_frame, tuple(right_eye_center), 2, (0, 255, 0), -1)

#         # PASO 3: Grabamos un video con lo que vemos de la camara (pulsando tecla G)
#         if cv2.waitKey(1) == ord("g"):
#             if not recording:
#                 # Obtener la fecha y hora actual
#                 now = datetime.now()
#                 timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
                
#                 # Crear el nombre del archivo de video
#                 video_name = f"videos/eyetracking-{timestamp}.mp4"
                
#                 # Crear el directorio si no existe
#                 os.makedirs("videos", exist_ok=True)
                
#                 # Inicializar el objeto VideoWriter
#                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                 out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
                
#                 print("Comenzando a grabar...")
#                 recording = True
#             else:
#                 # Detener la grabación
#                 out.release()
#                 out = None
#                 print(f"Grabación finalizada. Video guardado como: {video_name}")
#                 recording = False
        
#         # Escribir el frame en el objeto VideoWriter si se está grabando
#         if recording:
#             out.write(rotated_frame)

#         # Mostramos la imagen
#         cv2.imshow('liveStream' , rotated_frame)

#         if cv2.waitKey(1) == ord("q"):
#             break

# capture.release()
# cv2.destroyAllWindows()



#### Version 1.0 --- Seguimiento del ojo -- ERROR -- Sigue al ojo, no la mirada

# import cv2
# import dlib

# # Cargar el detector de rostros y el predictor de puntos faciales de Dlib
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Inicializar la captura de video desde la cámara IP
# capture = cv2.VideoCapture("http://192.168.1.51:4747/video")

# # Listas para almacenar las posiciones de los ojos
# left_eye_points = []
# right_eye_points = []

# while True:
#     ret, frame = capture.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         landmarks = predictor(gray, face)
        
#         # Detección de los ojos
#         left_eye = [landmarks.part(i) for i in range(36, 42)]
#         right_eye = [landmarks.part(i) for i in range(42, 48)]

#         # Calcular el centro de cada ojo
#         left_eye_center = (sum([point.x for point in left_eye]) // 6, sum([point.y for point in left_eye]) // 6)
#         right_eye_center = (sum([point.x for point in right_eye]) // 6, sum([point.y for point in right_eye]) // 6)
        
#         # Almacenar las posiciones de los ojos
#         left_eye_points.append(left_eye_center)
#         right_eye_points.append(right_eye_center)
        
#         # Dibujar los contornos de los ojos
#         for point in left_eye:
#             cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)
#         for point in right_eye:
#             cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

#         # Dibujar las trayectorias de los ojos
#         for point in left_eye_points:
#             cv2.circle(frame, point, 2, (0, 255, 0), -1)
#         for point in right_eye_points:
#             cv2.circle(frame, point, 2, (0, 0, 255), -1)
        
#         for i in range(1, len(left_eye_points)):
#             cv2.line(frame, left_eye_points[i-1], left_eye_points[i], (0, 255, 0), 1)
#         for i in range(1, len(right_eye_points)):
#             cv2.line(frame, right_eye_points[i-1], right_eye_points[i], (0, 0, 255), 1)

#     cv2.imshow('Eye Detection and Tracking', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()


#### Version 2.0 --- Seguimiento del ojo -- 

import cv2
import dlib
import numpy as np

# Cargar el detector de rostros y el predictor de puntos faciales de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializar la captura de video desde la cámara IP
cap = cv2.VideoCapture("http://192.168.1.51:4747/video")

# Función para calcular el centro de un conjunto de puntos
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Función para calcular la dirección de la mirada
def get_gaze_direction(eye_points, facial_landmarks):
    # Obtener el contorno del ojo
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    
    # Región del ojo
    eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    eye = cv2.resize(gray_eye, None, fx=5, fy=5)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        center = (x + int(w / 2), y + int(h / 2))
        return center

# Lista para almacenar los puntos de la dirección de la mirada
gaze_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Obtener la dirección de la mirada para cada ojo
        left_gaze = get_gaze_direction([36, 37, 38, 39, 40, 41], landmarks)
        right_gaze = get_gaze_direction([42, 43, 44, 45, 46, 47], landmarks)

        # Dibujar el punto de la dirección de la mirada en el marco
        if left_gaze is not None:
            gaze_points.append(left_gaze)
            cv2.circle(frame, left_gaze, 5, (0, 255, 0), -1)
        if right_gaze is not None:
            gaze_points.append(right_gaze)
            cv2.circle(frame, right_gaze, 5, (0, 0, 255), -1)
        
        # Opcional: Dibujar la trayectoria de la mirada
        for i in range(1, len(gaze_points)):
            cv2.line(frame, gaze_points[i-1], gaze_points[i], (255, 0, 0), 1)

    cv2.imshow('Gaze Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
