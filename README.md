# Proyecto de Inteligencia Artificial

Sistema de Reconocimiento Facial para detectar rostros y manipular una cámara mediante un servomotor

**Integrantes**

Ever Corazón 202042236

Olsend Luna 202040897


**Pasos para Instalar** (Se recomienda utilizar la versión 3.8.10 de Python)

1. Crear una carpeta llamada deteccion (puede ser cualquier nomnbre)
2. Abrir el cmd con de esa carpeta

Estando ubicados en el cmd de la carpeta, ejecutar los siguientes comandos
```
git clone https://github.com/EverCR1/Deteccion-de-Rostros-IA.git
```

Crear entorno
```
python -m venv venv
```

Activar entorno
```
venv\scripts\activate
```

Instalar dblib necesario para face recognition
```
pip install dlib-19.19.0-cp38-cp38-win_amd64.whl
```

Instalar librerías
```
python -m pip install -r requirements.txt
```

Comprobar librerías instaladas
```
pip freeze | pip list
```
