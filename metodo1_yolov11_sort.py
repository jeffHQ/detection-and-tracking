import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


COCO_A_VISDRONE = {
    0: 1,   # person -> pedestrian
    2: 4,   # car -> car
}


@dataclass
class PerfilSORT:
    mode: str
    model_path: str = "yolo11n.pt"
    imgsz: int = 1024
    conf: float = 0.20
    new_track_thr: float = 0.30
    nms_iou: float = 0.55
    iou_match: float = 0.35
    max_age: int = 15
    min_hits: int = 2
    max_det: int = 250
    min_side: float = 3.0
    min_area: float = 0.0


@dataclass
class Detection:
    xyxy: np.ndarray
    conf: float
    cls: int


class SortTrack:
    def __init__(self, track_id, det: Detection):
        self.track_id = track_id
        self.xyxy = det.xyxy.copy()
        self.conf = det.conf
        self.cls = det.cls
        self.velocity = np.zeros(4, dtype=np.float32)
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.class_votes = {det.cls: 1}

    def predict(self):
        self.xyxy = self.xyxy + self.velocity * 0.70
        self.age += 1
        self.time_since_update += 1

        if self.xyxy[2] <= self.xyxy[0] + 1:
            self.xyxy[2] = self.xyxy[0] + 1

        if self.xyxy[3] <= self.xyxy[1] + 1:
            self.xyxy[3] = self.xyxy[1] + 1

    def update(self, det: Detection):
        old_box = self.xyxy.copy()
        self.xyxy = det.xyxy.copy()
        self.velocity = det.xyxy - old_box
        self.conf = det.conf

        self.class_votes[det.cls] = self.class_votes.get(det.cls, 0) + 1
        self.cls = max(self.class_votes.items(), key=lambda x: x[1])[0]

        self.time_since_update = 0
        self.hits += 1


class SORTGeometrico:
    def __init__(self, iou_match, max_age, min_hits, new_track_thr):
        self.iou_match = iou_match
        self.max_age = max_age
        self.min_hits = min_hits
        self.new_track_thr = new_track_thr
        self.tracks = []
        self.next_id = 1

    def update(self, detecciones):
        for trk in self.tracks:
            trk.predict()

        matches, unmatched_tracks, unmatched_dets = self.asociar(detecciones)

        for ti, di in matches:
            self.tracks[ti].update(detecciones[di])

        for di in unmatched_dets:
            det = detecciones[di]

            if det.conf >= self.new_track_thr:
                self.crear_track(det)

        self.tracks = [
            trk for trk in self.tracks
            if trk.time_since_update <= self.max_age
        ]

        activos = [
            trk for trk in self.tracks
            if trk.time_since_update == 0 and trk.hits >= self.min_hits
        ]

        return activos

    def crear_track(self, det):
        self.tracks.append(
            SortTrack(
                track_id=self.next_id,
                det=det,
            )
        )
        self.next_id += 1

    def asociar(self, detecciones):
        if len(self.tracks) == 0:
            return [], [], list(range(len(detecciones)))

        if len(detecciones) == 0:
            return [], list(range(len(self.tracks))), []

        pares = []

        for ti, trk in enumerate(self.tracks):
            for di, det in enumerate(detecciones):
                if trk.cls != det.cls:
                    continue

                score_iou = iou(trk.xyxy, det.xyxy)

                if score_iou >= self.iou_match:
                    pares.append((score_iou, ti, di))

        pares.sort(reverse=True, key=lambda x: x[0])

        usados_t = set()
        usados_d = set()
        matches = []

        for score, ti, di in pares:
            if ti in usados_t or di in usados_d:
                continue

            usados_t.add(ti)
            usados_d.add(di)
            matches.append((ti, di))

        unmatched_tracks = [
            i for i in range(len(self.tracks))
            if i not in usados_t
        ]

        unmatched_dets = [
            i for i in range(len(detecciones))
            if i not in usados_d
        ]

        return matches, unmatched_tracks, unmatched_dets


def perfil_base():
    return PerfilSORT(
        mode="sort_base_conservative",
        imgsz=1024,
        conf=0.22,
        new_track_thr=0.32,
        nms_iou=0.55,
        iou_match=0.38,
        max_age=12,
        min_hits=2,
        max_det=180,
        min_side=4.0,
        min_area=20.0,
    )


def perfil_0182():
    return PerfilSORT(
        mode="sort_0182_small_balanced",
        imgsz=1024,
        conf=0.12,
        new_track_thr=0.22,
        nms_iou=0.55,
        iou_match=0.30,
        max_age=18,
        min_hits=2,
        max_det=260,
        min_side=3.0,
        min_area=0.0,
    )


def perfil_0268():
    return PerfilSORT(
        mode="sort_0268_highres_balanced",
        imgsz=1280,
        conf=0.14,
        new_track_thr=0.24,
        nms_iou=0.55,
        iou_match=0.30,
        max_age=18,
        min_hits=2,
        max_det=220,
        min_side=4.0,
        min_area=40.0,
    )


def perfil_0305():
    return PerfilSORT(
        mode="sort_0305_lowconf_filtered",
        imgsz=1280,
        conf=0.08,
        new_track_thr=0.18,
        nms_iou=0.55,
        iou_match=0.25,
        max_age=20,
        min_hits=2,
        max_det=280,
        min_side=8.0,
        min_area=100.0,
    )


def elegir_perfil(nombre_seq):
    if "uav0000182" in nombre_seq:
        return perfil_0182()

    if "uav0000268" in nombre_seq:
        return perfil_0268()

    if "uav0000305" in nombre_seq:
        return perfil_0305()

    return perfil_base()


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def iou_batch(box, boxes):
    if len(boxes) == 0:
        return np.array([], dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - inter

    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def clip_box(box, ancho, alto):
    box = box.copy()

    box[0] = max(0.0, min(float(box[0]), ancho - 1))
    box[1] = max(0.0, min(float(box[1]), alto - 1))
    box[2] = max(0.0, min(float(box[2]), ancho - 1))
    box[3] = max(0.0, min(float(box[3]), alto - 1))

    if box[2] <= box[0]:
        box[2] = box[0] + 1

    if box[3] <= box[1]:
        box[3] = box[1] + 1

    return box


def nms_por_clase(detecciones, thr):
    if not detecciones:
        return []

    final = []

    for cls in sorted(set(d.cls for d in detecciones)):
        grupo = [d for d in detecciones if d.cls == cls]
        grupo.sort(key=lambda d: d.conf, reverse=True)

        while grupo:
            best = grupo.pop(0)
            final.append(best)

            restantes = []

            for d in grupo:
                if iou(best.xyxy, d.xyxy) <= thr:
                    restantes.append(d)

            grupo = restantes

    final.sort(key=lambda d: d.conf, reverse=True)

    return final


def predicciones_yolo_a_detecciones(pred, img, perfil):
    alto, ancho = img.shape[:2]
    detecciones = []

    for b in pred.boxes:
        clase_coco = int(b.cls[0].item())

        if clase_coco not in COCO_A_VISDRONE:
            continue

        conf = float(b.conf[0].item())

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        box = clip_box(box, ancho, alto)

        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h

        if w < perfil.min_side or h < perfil.min_side:
            continue

        if area < perfil.min_area:
            continue

        detecciones.append(
            Detection(
                xyxy=box,
                conf=conf,
                cls=COCO_A_VISDRONE[clase_coco],
            )
        )

    detecciones = nms_por_clase(detecciones, perfil.nms_iou)

    return detecciones[:perfil.max_det]


def procesar_secuencia_sort(path_secuencia, path_salida_txt):
    nombre_seq = os.path.basename(path_secuencia)
    print(f"\n🚀 Procesando secuencia con YOLOv11 + SORT optimizado: {nombre_seq}")

    carpeta_img1 = os.path.join(path_secuencia, "img1")
    imagenes = sorted(glob.glob(os.path.join(carpeta_img1, "*.jpg")))

    if not imagenes:
        print("❌ No se encontraron imágenes.")
        return

    perfil = elegir_perfil(nombre_seq)

    print(
        f"   Perfil: {perfil.mode} | "
        f"model={perfil.model_path} | "
        f"imgsz={perfil.imgsz} | "
        f"conf={perfil.conf} | "
        f"new_track_thr={perfil.new_track_thr} | "
        f"iou_match={perfil.iou_match} | "
        f"max_age={perfil.max_age} | "
        f"min_hits={perfil.min_hits} | "
        f"max_det={perfil.max_det} | "
        f"min_side={perfil.min_side} | "
        f"min_area={perfil.min_area}"
    )

    model = YOLO(perfil.model_path)
    tracker = SORTGeometrico(
        iou_match=perfil.iou_match,
        max_age=perfil.max_age,
        min_hits=perfil.min_hits,
        new_track_thr=perfil.new_track_thr,
    )

    lineas_resultado_mot = []

    for idx, path_img in enumerate(imagenes):
        frame_id = idx + 1
        img = cv2.imread(path_img)

        if img is None:
            continue

        pred = model(
            img,
            conf=perfil.conf,
            imgsz=perfil.imgsz,
            verbose=False,
        )[0]

        detecciones = predicciones_yolo_a_detecciones(
            pred=pred,
            img=img,
            perfil=perfil,
        )

        tracks = tracker.update(detecciones)

        for trk in tracks:
            x1, y1, x2, y2 = trk.xyxy

            bb_left = max(0.0, float(x1))
            bb_top = max(0.0, float(y1))
            bb_width = max(1.0, float(x2 - x1))
            bb_height = max(1.0, float(y2 - y1))

            lineas_resultado_mot.append(
                f"{frame_id},{trk.track_id},"
                f"{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},"
                f"{trk.conf:.4f},{trk.cls},-1\n"
            )

    os.makedirs(os.path.dirname(path_salida_txt), exist_ok=True)

    with open(path_salida_txt, "w") as f:
        f.writelines(lineas_resultado_mot)

    print(f"✅ Resultados guardados en: {path_salida_txt}")