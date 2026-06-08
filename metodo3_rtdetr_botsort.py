import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import RTDETR


COCO_A_VISDRONE = {
    0: 1,   # person -> pedestrian
    2: 4,   # car -> car
}


@dataclass
class PerfilRTDETR:
    mode: str
    model_path: str = "rtdetr-l.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.70
    max_det: int = 200

    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.35
    track_buffer: int = 50
    match_thresh: float = 0.75

    min_side: float = 3.0
    min_area: float = 0.0


def perfil_base():
    return PerfilRTDETR(
        mode="rtdetr_botsort_base_fast",
        imgsz=640,
        conf=0.28,
        iou=0.70,
        max_det=180,
        track_high_thresh=0.35,
        track_low_thresh=0.10,
        new_track_thresh=0.35,
        track_buffer=50,
        match_thresh=0.75,
        min_side=3,
        min_area=0,
    )


def perfil_0182():
    return PerfilRTDETR(
        mode="rtdetr_botsort_0182_small",
        imgsz=960,
        conf=0.16,
        iou=0.70,
        max_det=260,
        track_high_thresh=0.25,
        track_low_thresh=0.08,
        new_track_thresh=0.25,
        track_buffer=70,
        match_thresh=0.72,
        min_side=3,
        min_area=0,
    )


def perfil_0268():
    return PerfilRTDETR(
        mode="rtdetr_botsort_0268_highres",
        imgsz=960,
        conf=0.16,
        iou=0.70,
        max_det=220,
        track_high_thresh=0.24,
        track_low_thresh=0.08,
        new_track_thresh=0.24,
        track_buffer=90,
        match_thresh=0.72,
        min_side=4,
        min_area=40,
    )


def perfil_0305():
    return PerfilRTDETR(
        mode="rtdetr_botsort_0305_gmc",
        imgsz=960,
        conf=0.12,
        iou=0.70,
        max_det=260,
        track_high_thresh=0.20,
        track_low_thresh=0.06,
        new_track_thresh=0.20,
        track_buffer=100,
        match_thresh=0.70,
        min_side=8,
        min_area=100,
    )


def elegir_perfil(nombre_seq):
    if "uav0000182" in nombre_seq:
        return perfil_0182()

    if "uav0000268" in nombre_seq:
        return perfil_0268()

    if "uav0000305" in nombre_seq:
        return perfil_0305()

    return perfil_base()


def crear_tracker_yaml(perfil, path_yaml):
    os.makedirs(os.path.dirname(path_yaml), exist_ok=True)

    contenido = f"""tracker_type: botsort
track_high_thresh: {perfil.track_high_thresh}
track_low_thresh: {perfil.track_low_thresh}
new_track_thresh: {perfil.new_track_thresh}
track_buffer: {perfil.track_buffer}
match_thresh: {perfil.match_thresh}
fuse_score: True

# Global Motion Compensation.
# sparseOptFlow usa flujo óptico para compensar movimiento de cámara.
gmc_method: sparseOptFlow

# ReID desactivado para no volver demasiado lento el método.
# BoT-SORT sigue usando GMC + asociación geométrica.
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""

    with open(path_yaml, "w") as f:
        f.write(contenido)


def filtrar_y_mapear_tracks(resultado, perfil):
    if resultado.boxes is None or resultado.boxes.id is None:
        return []

    boxes = resultado.boxes
    salidas = []

    for i in range(len(boxes)):
        clase_coco = int(boxes.cls[i].item())

        if clase_coco not in COCO_A_VISDRONE:
            continue

        track_id = int(boxes.id[i].item())
        conf = float(boxes.conf[i].item())

        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h

        if w < perfil.min_side or h < perfil.min_side:
            continue

        if area < perfil.min_area:
            continue

        clase_vis = COCO_A_VISDRONE[clase_coco]

        salidas.append(
            {
                "track_id": track_id,
                "x1": max(0.0, float(x1)),
                "y1": max(0.0, float(y1)),
                "w": max(1.0, float(w)),
                "h": max(1.0, float(h)),
                "conf": conf,
                "class": clase_vis,
            }
        )

    return salidas


def procesar_secuencia_rtdetr_botsort(path_secuencia, path_salida_txt):
    nombre_seq = os.path.basename(path_secuencia)
    print(f"\n🚀 Procesando secuencia con RT-DETR + BoT-SORT/GMC optimizado: {nombre_seq}")

    carpeta_img1 = os.path.join(path_secuencia, "img1")
    imagenes = sorted(glob.glob(os.path.join(carpeta_img1, "*.jpg")))

    if not imagenes:
        print("❌ No se encontraron imágenes.")
        return

    perfil = elegir_perfil(nombre_seq)

    tracker_yaml = os.path.join(
        "outputs",
        "trackers",
        f"botsort_{perfil.mode}.yaml",
    )

    crear_tracker_yaml(perfil, tracker_yaml)

    print(
        f"   Perfil: {perfil.mode} | "
        f"model={perfil.model_path} | "
        f"imgsz={perfil.imgsz} | "
        f"conf={perfil.conf} | "
        f"max_det={perfil.max_det} | "
        f"track_high={perfil.track_high_thresh} | "
        f"new_track={perfil.new_track_thresh} | "
        f"match={perfil.match_thresh} | "
        f"buffer={perfil.track_buffer} | "
        f"GMC=sparseOptFlow"
    )

    model = RTDETR(perfil.model_path)

    lineas = []

    # Importante:
    # persist=True mantiene el estado de BoT-SORT entre frames.
    # Como el modelo se instancia dentro de cada secuencia, el tracker no se mezcla
    # entre videos.
    for idx, path_img in enumerate(imagenes):
        frame_id = idx + 1
        img = cv2.imread(path_img)

        if img is None:
            continue

        resultado = model.track(
            img,
            persist=True,
            tracker=tracker_yaml,
            conf=perfil.conf,
            iou=perfil.iou,
            imgsz=perfil.imgsz,
            classes=[0, 2],
            max_det=perfil.max_det,
            verbose=False,
        )[0]

        tracks = filtrar_y_mapear_tracks(resultado, perfil)

        for trk in tracks:
            lineas.append(
                f"{frame_id},{trk['track_id']},"
                f"{trk['x1']:.2f},{trk['y1']:.2f},{trk['w']:.2f},{trk['h']:.2f},"
                f"{trk['conf']:.4f},{trk['class']},-1\n"
            )

    os.makedirs(os.path.dirname(path_salida_txt), exist_ok=True)

    with open(path_salida_txt, "w") as f:
        f.writelines(lineas)

    print(f"✅ Resultados guardados en: {path_salida_txt}")