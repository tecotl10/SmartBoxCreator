import cv2
import numpy as np

# Cargar los nombres de las clases desde el archivo coco.names
with open("coco.names", "r") as f:
    class_names = f.read().splitlines()

# Cargar la red YOLO con la configuración y los pesos de YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_output_layers(net):
    """Obtener los nombres de las capas de salida de la red"""
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    
    # Comprobar si 'out_layers' necesita ser aplanado
    if hasattr(out_layers, 'flatten'):
        out_layers = out_layers.flatten()
    else:
        out_layers = [i[0] for i in out_layers]

    output_layer_names = [layer_names[i - 1] for i in out_layers]
    
    return output_layer_names


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """Dibujar la predicción en la imagen"""
    label = str(class_names[class_id])
    color = (255, 0, 0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, f"{label}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Iniciar la captura de video desde la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

    # Convertir la imagen a un blob
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.2
    nms_threshold = 0.2

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Asegurarse de que 'indices' es un array NumPy para poder llamar a 'flatten()'
    if isinstance(indices, tuple):
        # Si 'indices' es una tupla, se asume que es vacía (no se encontraron cajas)
        indices = []
    else:
        # Aplanar el array de 'indices' para asegurar compatibilidad
        indices = indices.flatten()

    for i in indices:
        i = int(i)
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    # Mostrar el resultado en una ventana
    cv2.imshow("SmartBoxCreator Detection", frame)

    # Romper el bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el dispositivo de captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
