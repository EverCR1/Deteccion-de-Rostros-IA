#------------------------------ Importamos las librerias ------------------------------
import cv2
import mediapipe as mp
import serial
import tkinter as tk

#----------------------------- Puerto Serial Configuracion ----------------------------
com = serial.Serial("COM5", 9600, write_timeout= 10)
d = 'd'
i = 'i'
p = 'p'


# Captura y procesamiento de imágenes
def procesar_imagenes():
    #------------------------------ Declaramos el detector --------------------------------
    detector = mp.solutions.face_detection
    dibujo = mp.solutions.drawing_utils

    #------------------------------ Realizamos VideoCaptura --------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #-------------------------------Empezamos el while True --------------------------------
    with detector.FaceDetection(min_detection_confidence=0.75) as rostros:
        while True:
            ret, frame = cap.read()

            #Aplicamos espejo a los frames
            frame = cv2.flip(frame,1)

            #Correccion de color
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #Detectamos los rostros
            resultado = rostros.process(rgb)

            #Si hay rostros entramos al if
            if resultado.detections:
                # Solo tomamos el primer rostro detectado
                primer_rostro = resultado.detections[0]

                dibujo.draw_detection(frame, primer_rostro, dibujo.DrawingSpec(color=(0,255,0),))

                # Extraemos las coordenadas del primer rostro
                x = primer_rostro.location_data.relative_bounding_box.xmin
                y = primer_rostro.location_data.relative_bounding_box.ymin
                ancho = primer_rostro.location_data.relative_bounding_box.width
                alto = primer_rostro.location_data.relative_bounding_box.height

                # Pasamos las coordenadas a pixeles
                al, an, c = frame.shape
                x_px, y_px = int(x * an), int(y * al)
                ancho_px, alto_px = int(ancho * an), int(alto * al)

                # Extraemos el punto central
                cx = x_px + ancho_px // 2
                cy = y_px + alto_px // 2

                # Mostramos un punto en el centro
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
                cv2.line(frame, (cx, 0), (cx, 480), (0, 0, 255), 2)

                # Extraes aquí las condiciones para mover el servo utilizando cx como referencia

            cv2.imshow("Camara", frame)
            t = cv2.waitKey(1) # Presionar la tecla Esc para terminar la ejecución
            if t == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def abrir_ventana_entrenar():
    # Abrir la ventana de entrenamiento
    print("Abriendo ventana de entrenamiento...")

def abrir_ventana_visionar():
    # Abrir la ventana de visión
    procesar_imagenes()

# Crear la ventana principal
root = tk.Tk()
root.title("Visión por Computadora")

# Crear etiqueta para el título
titulo_label = tk.Label(root, text="Visión por Computadora", font=("Arial", 18))
titulo_label.pack(pady=20)

# Crear botón para entrenar
entrenar_button = tk.Button(root, text="Entrenar", width=20, command=abrir_ventana_entrenar)
entrenar_button.pack(pady=10)

# Crear botón para visionar
visionar_button = tk.Button(root, text="Visionar", width=20, command=abrir_ventana_visionar)
visionar_button.pack(pady=10)

# Ejecutar el bucle principal de la ventana
root.mainloop()