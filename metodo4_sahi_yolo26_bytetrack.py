import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


COCO_A_VISDRONE = {
    0: 1,   # person -> pedestrian
    2: 4,   # car -> car
}


@dataclass
class Perfil:
    mode: str
    imgsz: int = 1024
    yolo_conf: float = 0.10
    sahi_conf: float = 0.12
    slice_size: int = 768
    overlap: float = 0.20
    stride: int = 999999
    warmup: int = 0
    track_thr: float = 0.25
    match_thr: float = 0.95
    buffer: int = 50
    use_sahi: bool = False
    fuse_iou: float = 0.55
    max_det: int = 300


def perfil_base():
    return Perfil(
        mode="base",
        imgsz=1024,
        yolo_conf=0.10,
        track_thr=0.25,
        match_thr=0.95,
        buffer=50,
        use_sahi=False,
        fuse_iou=0.55,
        max_det=250,
    )


def perfil_sahi_0182():
    return Perfil(
        mode="sahi_0182_final",
        imgsz=1024,
        yolo_conf=0.09,
        sahi_conf=0.15,
        slice_size=512,
        overlap=0.25,
        stride=4,
        warmup=10,
        track_thr=0.22,
        match_thr=0.96,
        buffer=70,
        use_sahi=True,
        fuse_iou=0.48,
        max_det=260,
    )


def perfil_sahi_0268():
    return Perfil(
        mode="sahi_0268_final_960",
        imgsz=1280,
        yolo_conf=0.08,
        sahi_conf=0.11,
        slice_size=960,
        overlap=0.20,
        stride=4,
        warmup=12,
        track_thr=0.20,
        match_thr=0.90,
        buffer=90,
        use_sahi=True,
        fuse_iou=0.50,
        max_det=220,
    )


def perfil_sahi_0305():
    return Perfil(
        mode="sahi_0305_final",
        imgsz=1280,
        yolo_conf=0.07,
        sahi_conf=0.09,
        slice_size=768,
        overlap=0.20,
        stride=4,
        warmup=8,
        track_thr=0.16,
        match_thr=0.88,
        buffer=100,
        use_sahi=True,
        fuse_iou=0.48,
        max_det=250,
    )


def elegir_perfil(nombre_seq):
    if "uav0000182" in nombre_seq:
        return perfil_sahi_0182()

    if "uav0000268" in nombre_seq:
        return perfil_sahi_0268()

    if "uav0000305" in nombre_seq:
        return perfil_sahi_0305()

    return perfil_base()


def yolo_a_sv(pred):
    boxes = []
    confs = []
    clases = []

    for b in pred.boxes:
        cls = int(b.cls[0].item())

        if cls not in COCO_A_VISDRONE:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        confs.append(float(b.conf[0].item()))
        clases.append(COCO_A_VISDRONE[cls])

    if not boxes:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(clases, dtype=int),
    )


def sahi_a_sv(pred):
    boxes = []
    confs = []
    clases = []

    for p in pred.object_prediction_list:
        cls = int(p.category.id)

        if cls not in COCO_A_VISDRONE:
            continue

        x1, y1, x2, y2 = [float(v) for v in p.bbox.to_xyxy()]

        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        boxes.append([x1, y1, x2, y2])
        confs.append(float(p.score.value))
        clases.append(COCO_A_VISDRONE[cls])

    if not boxes:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(clases, dtype=int),
    )


def unir(a, b):
    if len(a) == 0:
        return b

    if len(b) == 0:
        return a

    return sv.Detections(
        xyxy=np.concatenate([a.xyxy, b.xyxy]),
        confidence=np.concatenate([a.confidence, b.confidence]),
        class_id=np.concatenate([a.class_id, b.class_id]),
    )


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


def nms_clase(det, thr):
    if len(det) == 0:
        return det

    boxes = det.xyxy
    scores = det.confidence
    clases = det.class_id

    keep = []

    for c in np.unique(clases):
        idx = np.where(clases == c)[0]
        orden = idx[np.argsort(scores[idx])[::-1]]

        while len(orden) > 0:
            actual = orden[0]
            keep.append(actual)

            if len(orden) == 1:
                break

            resto = orden[1:]
            ious = iou_batch(boxes[actual], boxes[resto])
            orden = resto[ious <= thr]

    keep = np.array(keep, dtype=int)

    return sv.Detections(
        xyxy=boxes[keep],
        confidence=scores[keep],
        class_id=clases[keep],
    )


def limitar(det, max_det):
    if len(det) <= max_det:
        return det

    idx = np.argsort(det.confidence)[::-1][:max_det]

    return sv.Detections(
        xyxy=det.xyxy[idx],
        confidence=det.confidence[idx],
        class_id=det.class_id[idx],
    )


def detectar_frame(img, frame_id, yolo_model, sahi_model, perfil):
    pred_yolo = yolo_model(
        img,
        conf=perfil.yolo_conf,
        imgsz=perfil.imgsz,
        verbose=False,
    )[0]

    det = yolo_a_sv(pred_yolo)

    usar_sahi = (
        perfil.use_sahi
        and sahi_model is not None
        and (
            frame_id <= perfil.warmup
            or frame_id % perfil.stride == 1
        )
    )

    if usar_sahi:
        pred_sahi = get_sliced_prediction(
            img,
            sahi_model,
            slice_height=perfil.slice_size,
            slice_width=perfil.slice_size,
            overlap_height_ratio=perfil.overlap,
            overlap_width_ratio=perfil.overlap,
            perform_standard_pred=False,

            # Esto quita el warning de SAHI.
            # Antes usábamos GREEDYNMM + IOS, pero con sahi_conf bajo
            # SAHI lo cambiaba automáticamente a NMS + IOU.
            # Ahora lo dejamos explícito.
            postprocess_type="NMS",
            postprocess_match_metric="IOU",

            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=False,
            verbose=0,
        )

        det_sahi = sahi_a_sv(pred_sahi)
        det = nms_clase(unir(det, det_sahi), perfil.fuse_iou)

    return limitar(det, perfil.max_det)


def crear_sahi_model(perfil):
    if not perfil.use_sahi:
        return None

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo26n.pt",
        confidence_threshold=perfil.sahi_conf,
        image_size=perfil.imgsz,
        device="cpu",
    )


def procesar_secuencia_sahi_bytetrack(path_secuencia, path_salida_txt):
    nombre_seq = os.path.basename(path_secuencia)
    print(f"\n🚀 Procesando secuencia con SAHI FINAL: {nombre_seq}")

    carpeta_img1 = os.path.join(path_secuencia, "img1")
    imagenes = sorted(glob.glob(os.path.join(carpeta_img1, "*.jpg")))

    if not imagenes:
        print("❌ No se encontraron imágenes.")
        return

    perfil = elegir_perfil(nombre_seq)

    print(
        f"   Perfil: {perfil.mode} | "
        f"imgsz={perfil.imgsz} | "
        f"yolo_conf={perfil.yolo_conf} | "
        f"sahi_conf={perfil.sahi_conf} | "
        f"SAHI={perfil.use_sahi} | "
        f"slice={perfil.slice_size} | "
        f"stride={perfil.stride} | "
        f"warmup={perfil.warmup} | "
        f"track_thr={perfil.track_thr} | "
        f"match_thr={perfil.match_thr} | "
        f"buffer={perfil.buffer}"
    )

    yolo_model = YOLO("yolo26n.pt")
    sahi_model = crear_sahi_model(perfil)

    tracker = sv.ByteTrack(
        track_activation_threshold=perfil.track_thr,
        lost_track_buffer=perfil.buffer,
        minimum_matching_threshold=perfil.match_thr,
        frame_rate=30,
    )

    id_a_clase = {}
    lineas = []

    for idx, path_img in enumerate(imagenes):
        frame_id = idx + 1
        img = cv2.imread(path_img)

        if img is None:
            continue

        det = detectar_frame(
            img=img,
            frame_id=frame_id,
            yolo_model=yolo_model,
            sahi_model=sahi_model,
            perfil=perfil,
        )

        tracks = tracker.update_with_detections(det)

        for i in range(len(tracks)):
            track_id = int(tracks.tracker_id[i])

            if tracks.class_id is not None:
                id_a_clase[track_id] = int(tracks.class_id[i])

            clase = id_a_clase.get(track_id, 4)

            x1, y1, x2, y2 = tracks.xyxy[i]

            left = max(0.0, float(x1))
            top = max(0.0, float(y1))
            width = max(1.0, float(x2 - x1))
            height = max(1.0, float(y2 - y1))

            conf = float(tracks.confidence[i]) if tracks.confidence is not None else 1.0

            lineas.append(
                f"{frame_id},{track_id},"
                f"{left:.2f},{top:.2f},{width:.2f},{height:.2f},"
                f"{conf:.4f},{clase},-1\n"
            )

    os.makedirs(os.path.dirname(path_salida_txt), exist_ok=True)

    with open(path_salida_txt, "w") as f:
        f.writelines(lineas)

    print(f"✅ Resultados guardados en: {path_salida_txt}")