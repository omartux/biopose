import cv2 as cv
import numpy as np
import pandas as pd

# Definir BODY_PARTS y POSE_PAIRS según el modelo utilizado
BODY_PARTS = { 
    "Nose": 0, "Neck": 1, "Right Shoulder": 2, "Right Elbow": 3, "Right Wrist": 4, 
    "Left Shoulder": 5, "Left Elbow": 6, "Left Wrist": 7, "Right Hip": 8, "Right Knee": 9, 
    "Right Ankle": 10, "Left Hip": 11, "Left Knee": 12, "Left Ankle": 13, 
    "Right Eye": 14, "Left Eye": 15, "Right Ear": 16, "Left Ear": 17, "Background": 18 
}
POSE_PAIRS = [
    ("Neck", "Right Shoulder"), ("Neck", "Left Shoulder"), ("Right Shoulder", "Right Elbow"),
    ("Right Elbow", "Right Wrist"), ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
    ("Neck", "Right Hip"), ("Right Hip", "Right Knee"), ("Right Knee", "Right Ankle"),
    ("Neck", "Left Hip"), ("Left Hip", "Left Knee"), ("Left Knee", "Left Ankle"),
    ("Neck", "Nose"), ("Nose", "Right Eye"), ("Right Eye", "Right Ear"),
    ("Nose", "Left Eye"), ("Left Eye", "Left Ear")
]

# Cargar el modelo previamente entrenado
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")  # Ajusta la ruta según tu modelo
inWidth = 368
inHeight = 368
thr = 0.2

# Inicializar captura de video
cap = cv.VideoCapture('salida3.mp4')
cap.set(3, 800)
cap.set(4, 800)

if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open video")

# Listas para almacenar las coordenadas de las articulaciones
all_joint_coordinates = []

frame_counter = 0
velocidad = 30
conti = 0

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    frame_counter += 1

    if not hasFrame:
        break

    if frame_counter == cap.get(cv.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    if conti == 0:
        print(frameWidth, frameHeight)
        conti = 1

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)

    # Guardar las coordenadas en la lista
    joint_coordinates = []
    for part in BODY_PARTS:
        idx = BODY_PARTS[part]
        if points[idx] is not None:
            joint_coordinates.extend([points[idx][0], points[idx][1]])
        else:
            joint_coordinates.extend([None, None])
    all_joint_coordinates.append(joint_coordinates)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Video Tesis', frame)

cap.release()
cv.destroyAllWindows()

# Guardar las coordenadas en un archivo CSV
columns = []
for part in BODY_PARTS:
    columns.append(part + '_x')
    columns.append(part + '_y')

df = pd.DataFrame(all_joint_coordinates, columns=columns)
df.to_csv('joint_coordinates.csv', index=False)

print("Las coordenadas de las articulaciones se han guardado en joint_coordinates.csv")
