import os
import time
import numpy as np
import pandas as pd
import motmetrics as mm

# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades
# ══════════════════════════════════════════════════════════════════════════════

def cargar_gt(path_txt: str) -> pd.DataFrame:
    """GT tiene clase -1, no filtramos por clase."""
    cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
            "conf", "class", "vis", "extra"]
    df = pd.read_csv(path_txt, header=None, names=cols)
    return df

def cargar_pred(path_txt: str) -> pd.DataFrame:
    """Predicciones sí tienen clase, filtramos solo 1 y 4."""
    cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
            "conf", "class", "vis"]
    df = pd.read_csv(path_txt, header=None, names=cols)
    df = df[df["class"].isin([1, 4])].reset_index(drop=True)
    return df


def df_a_dict_por_frame(df: pd.DataFrame) -> dict:
    """
    Convierte el DataFrame a:
        { frame_id: { obj_id: [bb_left, bb_top, bb_width, bb_height] } }
    """
    resultado = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        oid = int(row["id"])
        bbox = [row["bb_left"], row["bb_top"], row["bb_width"], row["bb_height"]]
        resultado.setdefault(f, {})[oid] = bbox
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluación principal
# ══════════════════════════════════════════════════════════════════════════════

def evaluar(
    path_gt: str,
    path_pred: str,
    tiempo_proceso_seg: float | None = None,
    n_frames: int | None = None,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Calcula MOTA, IDF1, ID Switches y FPS comparando GT vs predicciones.

    Parámetros
    ----------
    path_gt               : ruta al gt.txt (gt_mot_challenge)
    path_pred             : ruta al archivo .txt generado por tu método
    tiempo_proceso_seg    : segundos que tardó procesar la secuencia (para FPS)
    n_frames              : número total de frames de la secuencia (para FPS)
    iou_threshold         : umbral IoU para considerar una detección como TP (default 0.5)

    Retorna
    -------
    dict con MOTA, IDF1, IDS, FPS
    """
    print(f"\n📊 Evaluando:")
    print(f"   GT  : {path_gt}")
    print(f"   Pred: {path_pred}")

    gt_df   = cargar_gt(path_gt)
    pred_df = cargar_pred(path_pred)

    gt_por_frame   = df_a_dict_por_frame(gt_df)
    pred_por_frame = df_a_dict_por_frame(pred_df)

    todos_los_frames = sorted(
        set(gt_por_frame.keys()) | set(pred_por_frame.keys())
    )

    # Acumulador de motmetrics
    acc = mm.MOTAccumulator(auto_id=True)

    for frame_id in todos_los_frames:
        gt_frame   = gt_por_frame.get(frame_id, {})
        pred_frame = pred_por_frame.get(frame_id, {})

        gt_ids   = list(gt_frame.keys())
        pred_ids = list(pred_frame.keys())

        # Matriz de distancias IoU entre GT y predicciones
        # motmetrics espera distancia = 1 - IoU
        if gt_ids and pred_ids:
            dist_matrix = np.zeros((len(gt_ids), len(pred_ids)))
            for i, gid in enumerate(gt_ids):
                for j, pid in enumerate(pred_ids):
                    dist_matrix[i, j] = 1.0 - calcular_iou(
                        gt_frame[gid], pred_frame[pid]
                    )
        else:
            dist_matrix = np.empty((len(gt_ids), len(pred_ids)))

        acc.update(
            gt_ids,
            pred_ids,
            dist_matrix,
            # frameid=frame_id,
        )

    # Calcular métricas
    mh      = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches"],
        name="metodo",
    )

    mota = float(summary["mota"].iloc[0])
    idf1 = float(summary["idf1"].iloc[0])
    ids  = int(summary["num_switches"].iloc[0])

    # FPS
    fps = None
    if tiempo_proceso_seg and n_frames:
        fps = round(n_frames / tiempo_proceso_seg, 2)

    return {"mota": mota, "idf1": idf1, "id_switches": ids, "fps": fps}


def calcular_iou(bbox_a: list, bbox_b: list) -> float:
    """
    IoU entre dos bounding boxes en formato [x, y, w, h].
    """
    ax1, ay1 = bbox_a[0], bbox_a[1]
    ax2, ay2 = ax1 + bbox_a[2], ay1 + bbox_a[3]

    bx1, by1 = bbox_b[0], bbox_b[1]
    bx2, by2 = bx1 + bbox_b[2], by1 + bbox_b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = bbox_a[2] * bbox_a[3]
    area_b = bbox_b[2] * bbox_b[3]
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0