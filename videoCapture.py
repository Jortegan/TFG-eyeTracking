import cv2
import numpy as np
import os
from datetime import datetime
import dlib

# Inicializar el detector de caras y el predictor de puntos de referencia faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

capture = cv2.VideoCapture("http://172.20.10.7:4747/video")

if not capture.isOpened():
    print("Error al abrir el video.")
else:
    # Variables que usaremos
    rotation_angle = 0  # Declaramos la variable de la rotacion, por defecto, es como salga la imagen
    recording = False   # Modo de grabado, por defecto está apagado
    out = None          # Salida del modo de grabacion, por defecto desactivado

    while True:
        ret, frame = capture.read()

        # PASO 1: Comprobamos que tenemos acceso a la camara IP
        if not ret:
            print("No se pudo leer el cuadro del video.")
            break

        # PASO 2: Rotamos la imagen para que se adapte a nuestra pantalla (pulsando tecla R)
        if cv2.waitKey(1) == ord("r"):
            rotation_angle += 90
            if rotation_angle == 360:
                rotation_angle = 0

        # Obtener dimensiones de la imagen
        height, width = frame.shape[:2]
        
        # Calcular el centro de la imagen
        center = (width / 2, height / 2)
        
        # Definir la matriz de rotación
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        
        # Aplicar la transformación de rotación
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

        # Convertir la imagen a escala de grises para la detección facial
        gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras en la imagen
        faces = detector(gray)

        # Iterar sobre las caras detectadas
        for face in faces:
            # Obtener los puntos de referencia faciales
            landmarks = predictor(gray, face)

            # Los puntos 36-41 son para el ojo izquierdo y los puntos 42-47 son para el ojo derecho
            left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            # Calcular el centro de los ojos izquierdo y derecho
            left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
            right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

            # Dibujar un círculo en el centro de los ojos
            cv2.circle(rotated_frame, tuple(left_eye_center), 2, (0, 255, 0), -1)
            cv2.circle(rotated_frame, tuple(right_eye_center), 2, (0, 255, 0), -1)

        # PASO 3: Grabamos un video con lo que vemos de la camara (pulsando tecla G)
        if cv2.waitKey(1) == ord("g"):
            if not recording:
                # Obtener la fecha y hora actual
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
                
                # Crear el nombre del archivo de video
                video_name = f"videos/eyetracking-{timestamp}.mp4"
                
                # Crear el directorio si no existe
                os.makedirs("videos", exist_ok=True)
                
                # Inicializar el objeto VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
                
                print("Comenzando a grabar...")
                recording = True
            else:
                # Detener la grabación
                out.release()
                out = None
                print(f"Grabación finalizada. Video guardado como: {video_name}")
                recording = False
        
        # Escribir el frame en el objeto VideoWriter si se está grabando
        if recording:
            out.write(rotated_frame)

        # Mostramos la imagen
        cv2.imshow('liveStream' , rotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()

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

#             # Calcular la dirección de la mirada
#             gaze_direction = (right_eye_center - left_eye_center) / np.linalg.norm(right_eye_center - left_eye_center)
#             gaze_direction *= 50  # Ajustar la longitud del vector de dirección
            
#             # Convertir las coordenadas de la dirección de la mirada a enteros
#             gaze_direction = gaze_direction.astype(int)
            
#             # Dibujar la dirección de la mirada
#             cv2.arrowedLine(rotated_frame, tuple(left_eye_center), tuple(left_eye_center + gaze_direction), (0, 0, 255), 2)
#             cv2.arrowedLine(rotated_frame, tuple(right_eye_center), tuple(right_eye_center + gaze_direction), (0, 0, 255), 2)

#             # Mostrar la dirección de la mirada en la pantalla
#             gaze_text = f"Gaze Direction: {gaze_direction}"
#             cv2.putText(rotated_frame, gaze_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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

