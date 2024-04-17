import cv2

capture = cv2.VideoCapture("http://192.168.1.51:4747/video")

if not capture.isOpened():
    print("Error al abrir el video.")
else:
    while True:
        ret, frame = capture.read()

        if not ret:
            print("No se pudo leer el cuadro del video.")
            break

        cv2.imshow('liveStream' , frame)

        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
