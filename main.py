#--------------------------------------Importamos librerias--------------------------------------------
from tkinter import *
from tkinter.font import Font
import os
import cv2
import mediapipe as mp
import serial
import face_recognition

# Configuración del Puerto Serial (COM es el puerto al que se conecta el microcontrolador) y Variables
com = serial.Serial("COM3", 9600, write_timeout=10)
d = 'd'
i = 'i'
p = 'p'

numCam = 1 # Número de la cámara con la que ejecutaremos el programa

#------------------------- Funciones para crear las carpetas si no existen ------------------------
def crear_carpeta_usuarios():
    if not os.path.exists("usuarios"):
        os.makedirs("usuarios")

def crear_carpeta_faces():
    if not os.path.exists("faces"):
        os.makedirs("faces")

#------------------------ Función para entrenar un nuevo rostro --------------------------------------
def entrenar_rostro(pantalla1):
    crear_carpeta_usuarios()
    
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Capturamos el rostro
    cap = cv2.VideoCapture(numCam)  # Elegimos la cámara con la que vamos a hacer la detección
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

    # Llamamos a la función extraction para extraer solo el rostro
    extraction()

    Label(pantalla1, text="Rostro entrenado con éxito", fg="green", font=("Montserrat", 11)).pack()

#------------------------- Función con IA para detectar rostros y controlar al servomotor------------------------
def comparation():

    # Creamos las carpetas si no existen
    crear_carpeta_usuarios()
    crear_carpeta_faces()

    mp_face_detection = mp.solutions.face_detection

    imageFacesPath = "faces"

    facesEncodings = []
    facesNames = []
    for file_name in os.listdir(imageFacesPath):
        # Lee las imágenes de los rostros y las convierte a RGB
        image = cv2.imread(imageFacesPath + "/" + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Codifica los rostros usando face_recognition
        f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
        facesEncodings.append(f_coding)
        facesNames.append(file_name.split(".")[0])

    # Leemos el video
    cap = cv2.VideoCapture(numCam, cv2.CAP_DSHOW)
    # Detector facial
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # DEtectamos los rostros en pantalla
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
                orig = frame.copy()
                # Detección de rostros usando OpenCV
                faces = faceClassif.detectMultiScale(frame, 1.1, 5)
                # Si se detectan rostros se inicializan las validaciones de acuerdo al entrenamiento
                if resultado.detections is not None:
                    # Tomamos solo el primer rostro detectado
                    detection = resultado.detections[0]
                    orig = frame.copy()
                    # Solo procesamos y mostramos el primer rostro detectado
                    if len(faces) > 0:  # Verificamos si se detectaron rostros
                        (x, y, w, h) = faces[0]  # Obtenemos las coordenadas del primer rostro
                        face = orig[y:y + h, x:x + w]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        # Codifica el rostro detectado
                        actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
                        # Compara el rostro con los rostros previamente codificados
                        result = face_recognition.compare_faces(facesEncodings, actual_face_encoding)
                        if True in result:
                            index = result.index(True)
                            name = facesNames[index]
                            color = (125, 220, 0)
                            enviar_senal(detection, frame) # Enviamos señales al servomotor
                        else:
                            name = "Desconocido"
                            print(name)
                            color = (50, 50, 255)
                        # Dibuja el rectángulo y texto en el frame
                        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, "", (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # Muestra el frame con la detección de rostros
                cv2.imshow("Frame", frame) 
                k = cv2.waitKey(1) & 0xFF  # Cuando oprimamos "Esc" rompe el video
                if k == 27:
                    break
            #if ret == False:
             #   break
            
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
    # print("X, Y: ", x, y)

    # Pasamos el ancho y el alto a pixeles
    x1, y1 = int(ancho * an), int(alto * al)

    # Extraemos el punto central
    cx = (x + (x + x1)) // 2
    cy = (y + (y + y1)) // 2
    # print("Centro: ", cx, cy)

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

#------------------------- Función con IA para extraer solo los rostros de la imagen ------------------------
def extraction():
    imagesPath = "usuarios"
    facesPath = "faces"

    crear_carpeta_usuarios
    crear_carpeta_faces()
    
    # Detector facial
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = len(os.listdir(facesPath))  # Contador para el nuevo índice de archivo
    for imageName in os.listdir(imagesPath):
        print(imageName)
        image = cv2.imread(os.path.join(imagesPath, imageName))
        faces = faceClassif.detectMultiScale(image, 1.1, 5)
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, (150, 150))
            # Verificar si el rostro ya ha sido extraído antes de guardarlo
            if not any(imageName in filename for filename in os.listdir(facesPath)):
                cv2.imwrite(os.path.join(facesPath, str(count) + ".jpg"), face)
                count += 1

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
    Button(frame, text="Controlar Cámara", height=2, width=30, font=montserrat_font, command=comparation).pack(pady=5)
    
    # Botón para entrenar
    Button(frame, text="Entrenar Nuevo Rostro", height=2, width=30, font=montserrat_font, command=lambda: entrenar_rostro(pantalla)).pack(pady=5)
    
    # Centrar la ventana en la pantalla
    pantalla.mainloop()

# Llamamos a la función principal para iniciar la pantalla
pantalla_principal()
