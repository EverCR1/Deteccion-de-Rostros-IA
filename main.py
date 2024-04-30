#--------------------------------------Importamos librerias--------------------------------------------
from tkinter import *
from tkinter.font import Font
import os
import cv2
import mediapipe as mp
import serial

#------------------------- Función para crear la carpeta de usuarios si no existe ------------------------
def crear_carpeta_usuarios():
    if not os.path.exists("usuarios"):
        os.makedirs("usuarios")

#------------------------ Función para entrenar un nuevo rostro --------------------------------------
def entrenar_rostro(pantalla1):
    crear_carpeta_usuarios()
    
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Capturamos el rostro
    cap = cv2.VideoCapture(0)  # Elegimos la cámara con la que vamos a hacer la detección
    with mp_face_detection.FaceDetection(min_detection_confidence=0.75) as rostros:
        while True:
            ret, frame = cap.read()  # Leemos el video

            # Aplicamos el efecto espejo a los frames
            frame = cv2.flip(frame, 1)

            # Guardamos una copia de la imagen original
            frame_original = frame.copy()

            # Corrección de color
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detección de los rostros
            resultado = rostros.process(rgb)

            # Si se detectan rostros se inicializan las validaciones de acuerdo al entrenamiento
            if resultado.detections is not None:
                for detection in resultado.detections:
                    mp_drawing.draw_detection(frame, detection)  # Dibujamos sobre el rostro detectado

                    # Dibujamos el recuadro sobre el rostro detectado
                    x, y, w, h = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1]), \
                                 int(detection.location_data.relative_bounding_box.ymin * frame.shape[0]), \
                                 int(detection.location_data.relative_bounding_box.width * frame.shape[1]), \
                                 int(detection.location_data.relative_bounding_box.height * frame.shape[0])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Entrenamiento', frame)  # Mostramos el video en pantalla
            if cv2.waitKey(1) == 27:  # Cuando oprimamos "Esc" rompe el video
                break

    usuario_img = str(len(os.listdir("usuarios")) + 1)  # Rostros de los usuarios entrenados almacenados en la carpeta
    cv2.imwrite(os.path.join("usuarios", usuario_img + ".jpg"), frame_original)  # Guardamos la imagen original
    cap.release()  # Cerramos
    cv2.destroyAllWindows()

    Label(pantalla1, text="Rostro entrenado con éxito", fg="green", font=("Montserrat", 11)).pack()

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

    # Revisamos si el punto clave 1 no es nulo para evitar errores
    if len(keypoints1) == 0:
        return 0.0

    # Calculamos la similitud como el número de coincidencias dividido por el total de coincidencias
    similitud = len(matches) / len(keypoints1)

    return similitud



# Configuración del Puerto Serial (COM5 es el puerto al que se conecta el microcontrolador)
com = serial.Serial("COM5", 9600, write_timeout=10)
d = 'd'
i = 'i'
p = 'p'

#------------------------- Función con IA para detectar rostros y controlar al servomotor------------------------
def control_servo():
    mp_face_detection = mp.solutions.face_detection

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.75) as rostros:
        while True:
            ret, frame = cap.read()

            # Aplicamos espejo a los frames
            frame = cv2.flip(frame, 1)

            # Corrección de color
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detección de los rostros
            resultado = rostros.process(rgb)

            # Si se detectan rostros se inicializan las validaciones de acuerdo al entrenamiento
            if resultado.detections is not None:
                # Tomamos solo el primer rostro detectado
                detection = resultado.detections[0]

                # Verificamos que el sistema conozca al rostro detectado
                x = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1])
                y = int(detection.location_data.relative_bounding_box.ymin * frame.shape[0])
                w = int(detection.location_data.relative_bounding_box.width * frame.shape[1])
                h = int(detection.location_data.relative_bounding_box.height * frame.shape[0])
                cara = frame[y:y+h, x:x+w]
                if not cara.size == 0:
                    cara = cv2.resize(cara, (150, 200))
                else:
                    print("Error: No se detectó ningún rostro.")
                    continue

                # Calculamos las coordenadas del centro de la frente del rostro
                punto_x = x + w // 2  # Coordenada x del centro del rostro
                punto_y = y + h // 8  # Coordenada y del centro del rostro

                # Dibujamos un círculo rojo sobre la frente
                cv2.circle(frame, (punto_x, punto_y), int(min(w, h) / 2), (0, 0, 255), 2)

                # # Dibujamos un punto en el centro de la frente
                cv2.circle(frame, (punto_x, punto_y), 8, (0, 255, 0), -1)

                # # Dibujamos una línea horizontal
                cv2.line(frame, (0, punto_y), (frame.shape[1], punto_y), (0, 0, 0), 2)

                # # Dibujamos una línea vertical 
                cv2.line(frame, (punto_x, 0), (punto_x, frame.shape[0]), (0, 0, 0), 2)

                # Obtenemos los rostros entrenados y validamos
                for usuario_img in os.listdir("usuarios"):
                    img_path = os.path.join("usuarios", usuario_img)
                    rostro_reg = cv2.imread(img_path)
                    similitud = comparar_rostros(cara, rostro_reg) # Comparamos rostros
                    if similitud >= 0.75:
                        print("El usuario coincide con el número:", usuario_img.split('.')[0]) # Verificar coincidencias
                        print(similitud)
                        enviar_senal(detection, frame) # Si el rostro coincide con alguno entrenado, enviar señales al servomotor
                        break
                else:
                    print("No se encontró coincidencia con ningún usuario entrenado.")

            cv2.imshow("Control de Servomotor", frame)
            t = cv2.waitKey(1)
            if t == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


#------------------------- Función para enviar señales al servomotor ------------------------
def enviar_senal(puntos, frame):
    # Extraemos el ancho y el alto del frame
    al, an, c = frame.shape

    # Extraemos el medio de la pantalla
    centro = int(an / 2)

    # Extraemos las coordenadas X e Y min
    x = puntos.location_data.relative_bounding_box.xmin
    y = puntos.location_data.relative_bounding_box.ymin

    # Extraemos el ancho y el alto
    ancho = puntos.location_data.relative_bounding_box.width
    alto = puntos.location_data.relative_bounding_box.height

    # Pasamos X e Y a coordenadas en pixeles
    x, y = int(x * an), int(y * al)
    print("X, Y: ", x, y)

    # Pasamos el ancho y el alto a pixeles
    x1, y1 = int(ancho * an), int(alto * al)

    # Extraemos el punto central
    cx = (x + (x + x1)) // 2
    cy = (y + (y + y1)) // 2
    # print("Centro: ", cx, cy)

    # Mostrar un punto en el centro
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
    cv2.line(frame, (cx, 0), (cx, 480), (0, 0, 255), 2)

    # Condicionales para mover el servo hacia cierta dirección
    if cx < centro - 50:
        # Movemos hacia la izquierda
        print("Izquierda")
        com.write(i.encode('ascii')) # Puerto Serial establecido
    elif cx > centro + 50:
        # Movemos hacia la derecha
        print("Derecha")
        com.write(d.encode('ascii'))
    elif cx == centro:
        # Detenemos el servo
        print("Parar")
        com.write(p.encode('ascii'))

#------------------------- Función de la pantalla principal ------------------------------------------------
def pantalla_principal():
    global pantalla  # Globalizamos la variable para usarla en otras funciones
    pantalla = Tk()
    pantalla.title("Sistema de Reconocimiento Facial")  # Asignamos el título de la pantalla

    montserrat_font = Font(family="Montserrat", size=12) # Fuente a utilizar
    
    # Obtenemos las dimensiones de la pantalla
    screen_width = pantalla.winfo_screenwidth()
    screen_height = pantalla.winfo_screenheight()
    
    # Calculamos las coordenadas para centrar la ventana en la pantalla
    x_position = (screen_width - 300) // 2
    y_position = (screen_height - 250) // 2
    
    # Configuramos la geometría de la ventana y la posicionamos en el centro de la pantalla
    pantalla.geometry("400x200+{}+{}".format(x_position, y_position))
    
     # Crear un marco para contener los elementos
    frame = Frame(pantalla, bg="gray", width=300, height=250)
    frame.pack(expand=True, fill="both")
    
    # Etiqueta para el título
    Label(frame, text="Control de Servomotor con IA", bg="gray", fg="white", font=montserrat_font, pady=10).pack()
    
    # Botón para controlar el servo
    Button(frame, text="Controlar Servo", height=2, width=30, font=montserrat_font, command=control_servo).pack(pady=5)
    
    # Botón para entrenar
    Button(frame, text="Entrenar", height=2, width=30, font=montserrat_font, command=lambda: entrenar_rostro(pantalla)).pack(pady=5)
    
    # Centrar la ventana en la pantalla
    pantalla.mainloop()

# Llamamos a la función principal para iniciar la pantalla
pantalla_principal()

