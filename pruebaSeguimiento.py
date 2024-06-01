#### Version 3.0 -- Seguimiento de la mirada -- Va bien -- Saca por pantalla donde se está mirando

# import cv2
# import dlib
# import numpy as np

# # Cargar el detector de caras y el predictor de puntos faciales
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Función para obtener la región de interés del ojo
# def get_eye_region(shape, eye_points):
#     points = [shape.part(point).x for point in eye_points] + [shape.part(point).y for point in eye_points]
#     return points

# # Función para detectar la posición de la pupila (simplificada)
# def detect_pupil(eye_image):
#     gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
#     _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
#     moments = cv2.moments(threshold_eye, False)
#     if moments['m00'] != 0:
#         cx = int(moments['m10'] / moments['m00'])
#         cy = int(moments['m01'] / moments['m00'])
#         return (cx, cy)
#     else:
#         return None

# # Inicializar la captura de video
# cap = cv2.VideoCapture("http://192.168.1.51:4747/video")
# previous_position = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         landmarks = predictor(gray, face)

#         left_eye_region = get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])
#         right_eye_region = get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])
        
#         left_eye = frame[min(left_eye_region[1::2]):max(left_eye_region[1::2]), min(left_eye_region[0::2]):max(left_eye_region[0::2])]
#         right_eye = frame[min(right_eye_region[1::2]):max(right_eye_region[1::2]), min(right_eye_region[0::2]):max(right_eye_region[0::2])]

#         left_pupil = detect_pupil(left_eye)
#         right_pupil = detect_pupil(right_eye)

#         if left_pupil and right_pupil:
#             avg_pupil_position = ((left_pupil[0] + right_pupil[0]) // 2, (left_pupil[1] + right_pupil[1]) // 2)
#             print(f'Pupil position: {avg_pupil_position}')
            
#             if previous_position is not None:
#                 if avg_pupil_position[1] < previous_position[1]:  # Check if the user is looking up
#                     cv2.line(frame, previous_position, avg_pupil_position, (0, 255, 0), 2)
            
#             previous_position = avg_pupil_position
    
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



#### Version 4.0 -- Seguimiento del ojo -- Pinta una nueva ventana donde se dibujan las lineas
##### Funciona correctamente, pero no utiliza toda la pantalla, se queda solo en la esquina superior.

# import cv2
# import dlib
# import numpy as np

# # Cargar el detector de caras y el predictor de puntos faciales
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Función para obtener la región de interés del ojo
# def get_eye_region(shape, eye_points):
#     points = [(shape.part(point).x, shape.part(point).y) for point in eye_points]
#     return points

# # Función para detectar la posición de la pupila (simplificada)
# def detect_pupil(eye_image):
#     gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
#     _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
#     moments = cv2.moments(threshold_eye, False)
#     if moments['m00'] != 0:
#         cx = int(moments['m10'] / moments['m00'])
#         cy = int(moments['m01'] / moments['m00'])
#         return (cx, cy)
#     else:
#         return None

# # Inicializar la captura de video
# cap = cv2.VideoCapture("http://192.168.1.51:4747/video")
# previous_position = None

# # Crear una ventana separada para dibujar el seguimiento de la mirada
# drawing_window = np.zeros((500, 500, 3), dtype=np.uint8)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         landmarks = predictor(gray, face)

#         left_eye_region = get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])
#         right_eye_region = get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])
        
#         left_eye = frame[min([p[1] for p in left_eye_region]):max([p[1] for p in left_eye_region]), 
#                          min([p[0] for p in left_eye_region]):max([p[0] for p in left_eye_region])]
#         right_eye = frame[min([p[1] for p in right_eye_region]):max([p[1] for p in right_eye_region]), 
#                           min([p[0] for p in right_eye_region]):max([p[0] for p in right_eye_region])]

#         left_pupil = detect_pupil(left_eye)
#         right_pupil = detect_pupil(right_eye)

#         if left_pupil and right_pupil:
#             avg_pupil_position = ((left_pupil[0] + right_pupil[0]) // 2, (left_pupil[1] + right_pupil[1]) // 2)
            
#             # Ajustar las coordenadas de la pupila al tamaño de la ventana de dibujo
#             scale_factor = 5  # Ajusta este factor según sea necesario
#             drawing_point = (avg_pupil_position[0] * scale_factor, avg_pupil_position[1] * scale_factor)
            
#             if previous_position is not None:
#                 cv2.line(drawing_window, previous_position, drawing_point, (0, 255, 0), 2)
            
#             previous_position = drawing_point
    
#     cv2.imshow('Frame', frame)
#     cv2.imshow('Gaze Tracking', drawing_window)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




#### Version 6.0 -- Mejora de muestreo de puntos -- Error -- No se muestran correctamente

import cv2
import dlib
import numpy as np

# Cargar el detector de caras y el predictor de puntos faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Dimensiones de la ventana de dibujo
drawing_window_height, drawing_window_width = 500, 500

# Función para obtener la región de interés del ojo
def get_eye_region(shape, eye_points):
    points = [(shape.part(point).x, shape.part(point).y) for point in eye_points]
    return points

# Función para detectar la posición de la pupila (simplificada)
def detect_pupil(eye_image):
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
    moments = cv2.moments(threshold_eye, False)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)
    else:
        return None

# Función para normalizar y escalar las coordenadas
def map_to_window(pupil_position, eye_region, eye_box):
    ex, ey, ew, eh = eye_box
    normalized_x = (pupil_position[0] + ex) / ew
    normalized_y = (pupil_position[1] + ey) / eh

    scaled_x = int(normalized_x * drawing_window_width)
    scaled_y = int(normalized_y * drawing_window_height)

    return (scaled_x, scaled_y)

# Inicializar la captura de video
cap = cv2.VideoCapture("http://192.168.1.51:4747/video")
previous_position = None

# Crear una ventana separada para dibujar el seguimiento de la mirada
drawing_window = np.zeros((drawing_window_height, drawing_window_width, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_region = get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])
        right_eye_region = get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])
        
        left_eye_box = (min([p[0] for p in left_eye_region]), min([p[1] for p in left_eye_region]), 
                        max([p[0] for p in left_eye_region]) - min([p[0] for p in left_eye_region]), 
                        max([p[1] for p in left_eye_region]) - min([p[1] for p in left_eye_region]))
        
        right_eye_box = (min([p[0] for p in right_eye_region]), min([p[1] for p in right_eye_region]), 
                         max([p[0] for p in right_eye_region]) - min([p[0] for p in right_eye_region]), 
                         max([p[1] for p in right_eye_region]) - min([p[1] for p in right_eye_region]))

        left_eye = frame[left_eye_box[1]:left_eye_box[1] + left_eye_box[3], 
                         left_eye_box[0]:left_eye_box[0] + left_eye_box[2]]
        right_eye = frame[right_eye_box[1]:right_eye_box[1] + right_eye_box[3], 
                          right_eye_box[0]:right_eye_box[0] + right_eye_box[2]]

        left_pupil = detect_pupil(left_eye)
        right_pupil = detect_pupil(right_eye)

        if left_pupil and right_pupil:
            avg_pupil_position = ((left_pupil[0] + right_pupil[0]) // 2, (left_pupil[1] + right_pupil[1]) // 2)
            
            # Mapear las coordenadas de la pupila a la ventana de dibujo
            drawing_point = map_to_window(avg_pupil_position, left_eye_region + right_eye_region, left_eye_box)
            
            if previous_position is not None:
                cv2.line(drawing_window, previous_position, drawing_point, (0, 255, 0), 2)
            
            previous_position = drawing_point
    
    cv2.imshow('Frame', frame)
    cv2.imshow('Gaze Tracking', drawing_window)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



