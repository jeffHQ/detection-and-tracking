import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

def procesar_secuencia_bytetrack(path_secuencia, path_salida_txt):
    print(f"\n🚀 Procesando secuencia: {os.path.basename(path_secuencia)}")

    model = YOLO("yolo26n.pt")
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )

    carpeta_img1 = os.path.join(path_secuencia, "img1")
    imagenes = sorted(glob.glob(os.path.join(carpeta_img1, "*.jpg")))
    if not imagenes:
        print("❌ No se encontraron imágenes.")
        return

    id_a_clase: dict[int, int] = {}
    lineas_resultado_mot = []

    for idx, path_img in enumerate(imagenes):
        frame_id = idx + 1
        img = cv2.imread(path_img)

        predicciones = model(img, conf=0.1, verbose=False)[0]

        boxes_xyxy = []
        confianzas = []
        clases_vis = []

        for box in predicciones.boxes:
            clase_coco = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if clase_coco == 0: clase_visdrone = 1
            elif clase_coco == 2: clase_visdrone = 4
            else: continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes_xyxy.append([x1, y1, x2, y2])
            confianzas.append(conf)
            clases_vis.append(clase_visdrone)

        if len(boxes_xyxy) == 0:
            tracker.update_with_detections(sv.Detections.empty())
            continue

        detecciones_sv = sv.Detections(
            xyxy = np.array(boxes_xyxy, dtype=np.float32),
            confidence = np.array(confianzas, dtype=np.float32),
            class_id = np.array(clases_vis, dtype=int),
        )

        tracks = tracker.update_with_detections(detecciones_sv)

        for i in range(len(tracks)):
            track_id = int(tracks.tracker_id[i])

            if tracks.class_id is not None:
                id_a_clase[track_id] = int(tracks.class_id[i])
            clase_mot = id_a_clase.get(track_id, 4)

            x1, y1, x2, y2 = tracks.xyxy[i]
            bb_left = max(0.0, float(x1))
            bb_top = max(0.0, float(y1))
            bb_width = max(1.0, float(x2 - x1))
            bb_height = max(1.0, float(y2 - y1))
            conf_out = float(tracks.confidence[i]) if tracks.confidence is not None else 1.0

            linea = (
                f"{frame_id},{track_id},"
                f"{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},"
                f"{conf_out:.4f},{clase_mot},-1\n"
            )
            lineas_resultado_mot.append(linea)

    os.makedirs(os.path.dirname(path_salida_txt), exist_ok=True)
    with open(path_salida_txt, "w") as f:
        f.writelines(lineas_resultado_mot)
    print(f"✅ Resultados guardados en: {path_salida_txt}")