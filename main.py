#--------------------------------------Importamos librerias--------------------------------------------

from tkinter import *
import os
import cv2
import mediapipe as mp
import serial

#------------------------- Función para crear la carpeta de usuarios si no existe ------------------------

def crear_carpeta_usuarios():
    if not os.path.exists("usuarios"):
        os.makedirs("usuarios")

#------------------------ Función para almacenar el registro facial --------------------------------------
def registro_facial(pantalla1):
    crear_carpeta_usuarios()
    # Vamos a capturar el rostro
    cap = cv2.VideoCapture(0)  # Elegimos la cámara con la que vamos a hacer la detección
    while True:
        ret, frame = cap.read()  # Leemos el video

        # Aplicamos el efecto espejo a los frames
        frame = cv2.flip(frame, 1)

        # Corrección de color
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('Registro Facial', frame)  # Mostramos el video en pantalla
        if cv2.waitKey(1) == 27:  # Cuando oprimamos "Escape" rompe el video
            break

    usuario_img = str(len(os.listdir("usuarios")) + 1)  # Nombre de usuario basado en la cantidad de archivos en la carpeta
    cv2.imwrite(os.path.join("usuarios", usuario_img + ".jpg"), frame)  # Guardamos la última captura del video como imagen
    cap.release()  # Cerramos
    cv2.destroyAllWindows()

    Label(pantalla1, text="Registro Facial Exitoso", fg="green", font=("Calibri", 11)).pack()


#------------------------- Función para comparar los rostros ------------------------
def comparar_rostros(rostro1, rostro2):
    # Convertimos las imágenes a escala de grises
    rostro1_gris = cv2.cvtColor(rostro1, cv2.COLOR_BGR2GRAY)
    rostro2_gris = cv2.cvtColor(rostro2, cv2.COLOR_BGR2GRAY)

    # Inicializamos el detector ORB
    orb = cv2.ORB_create()

    # Detectamos los puntos clave y los descriptores en ambos rostros
    keypoints1, descriptors1 = orb.detectAndCompute(rostro1_gris, None)
    keypoints2, descriptors2 = orb.detectAndCompute(rostro2_gris, None)

    # Inicializamos el comparador de coincidencias de fuerza bruta
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Realizamos la comparación de los descriptores
    matches = bf.match(descriptors1, descriptors2)

    # Ordenamos las coincidencias por distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Check if keypoints1 is empty to avoid ZeroDivisionError
    if len(keypoints1) == 0:
        return 0.0

    # Calculamos la similitud como el número de coincidencias dividido por el total de coincidencias
    similitud = len(matches) / len(keypoints1)

    return similitud



# Configuración del Puerto Serial
com = serial.Serial("COM5", 9600, write_timeout=10)
d = 'd'
i = 'i'
p = 'p'

# Controlar al servomotor
def control_servo():

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.75) as rostros:
        while True:
            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resultado = rostros.process(rgb)

            if resultado.detections is not None:
                for detection in resultado.detections:
                    mp_drawing.draw_detection(frame, detection)

                    for detection in resultado.detections:
                        x, y, w, h = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1]), int(detection.location_data.relative_bounding_box.ymin * frame.shape[0]), int(detection.location_data.relative_bounding_box.width * frame.shape[1]), int(detection.location_data.relative_bounding_box.height * frame.shape[0])
                        cara = frame[y:y+h, x:x+w]
                        if not cara.size == 0:
                            cara = cv2.resize(cara, (150, 200))
                        else:
                            print("Error: No se detectó ningún rostro.")
                            continue  # Skip further processing for this frame

                        for usuario_img in os.listdir("usuarios"):
                            img_path = os.path.join("usuarios", usuario_img)
                            rostro_reg = cv2.imread(img_path)
                            similitud = comparar_rostros(cara, rostro_reg)
                            if similitud >= 0.60:
                                print("El usuario coincide con el número:", usuario_img.split('.')[0])
                                enviar_senal(detection, frame)
                                break
                        else:
                            print("No se encontró coincidencia con ningún usuario registrado")

            cv2.imshow("Camara", frame)
            t = cv2.waitKey(1)
            if t == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# Enviar señal al servomotor
def enviar_senal(puntos, frame):
    #Extraemos el ancho y el alto del frame
    al, an, c = frame.shape

    #Extraemos el medio de la pantalla
    centro = int(an / 2)

    #Extraemos las coordenadas X e Y min
    x = puntos.location_data.relative_bounding_box.xmin
    y = puntos.location_data.relative_bounding_box.ymin

    #Extraemos el ancho y el alto
    ancho = puntos.location_data.relative_bounding_box.width
    alto = puntos.location_data.relative_bounding_box.height

    #Pasamos X e Y a coordenadas en pixeles
    x, y = int(x * an), int(y * al)
    print("X, Y: ", x, y)

    #Pasamos el ancho y el alto a pixeles
    x1, y1 = int(ancho * an), int(alto * al)

    #Extraemos el punto central
    cx = (x + (x + x1)) // 2
    cy = (y + (y + y1)) // 2
    #print("Centro: ", cx, cy)

    #Mostrar un punto en el centro
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
    cv2.line(frame, (cx, 0), (cx, 480), (0, 0, 255), 2)

    #Condiciones para mover el servo
    if cx < centro - 50:
        #Movemos hacia la izquierda
        print("Izquierda")
        com.write(i.encode('ascii'))
    elif cx > centro + 50:
        #Movemos hacia la derecha
        print("Derecha")
        com.write(d.encode('ascii'))
    elif cx == centro:
        #Paramos el servo
        print("Parar")
        com.write(p.encode('ascii'))

#------------------------- Función de la pantalla principal ------------------------------------------------

def pantalla_principal():
    global pantalla  # Globalizamos la variable para usarla en otras funciones
    pantalla = Tk()
    pantalla.geometry("300x250")  # Asignamos el tamaño de la ventana
    pantalla.title("Visión por Computadora")  # Asignamos el título de la pantalla
    Label(text="Login Inteligente", bg="gray", width="300", height="2", font=("Verdana", 13)).pack()  # Asignamos características de la ventana

     # Creamos los botones
    Button(text="Iniciar Sesión", height="2", width="30", command=control_servo).pack() 
    Button(text="Registrar Usuario", height="2", width="30", command=lambda: registro_facial(pantalla)).pack()
    
    pantalla.mainloop()

# Llamamos a la función principal para iniciar la pantalla
pantalla_principal()
