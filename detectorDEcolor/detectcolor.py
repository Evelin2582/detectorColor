import cv2
import numpy as np


color_ranges = {
    "rojo": ([0, 100, 100], [10, 255, 255]),
    "verde": ([35, 100, 100], [85, 255, 255]),
    "azul": ([100, 100, 100], [140, 255, 255]),
    "amarillo": ([20, 100, 100], [30, 255, 255]),
    "negro": ([0, 0, 0], [180, 255, 30]),
} 

def detectar_colores(frame):
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Iterar sobre cada rango de color
    for color, (lower, upper) in color_ranges.items():
        # Crear máscaras
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(hsv, lower_np, upper_np)

        # Encontrar contornos
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar los contornos
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 500:  # Filtrar por área mínima
                x, y, w, h = cv2.boundingRect(contorno)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    # Captura de video (0 es la cámara por defecto)
    cap = cv2.VideoCapture(0)

    while True:
        # Leer un frame
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar colores en el frame actual
        frame_con_colores = detectar_colores(frame)

        # Mostrar el frame procesado
        cv2.imshow('Detección de Colores', frame_con_colores)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
