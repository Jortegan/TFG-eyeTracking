import cv2
import numpy as np
import os
from datetime import datetime

capture = cv2.VideoCapture("http://192.168.1.67:4747/video")

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
