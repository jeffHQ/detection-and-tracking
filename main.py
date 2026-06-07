import time
import glob
import os

from metodo1_yolov11_sort import procesar_secuencia_sort
from metodo2_yolo26_bytetrack import procesar_secuencia_bytetrack
from metodo4_sahi_yolo26_bytetrack import procesar_secuencia_sahi_bytetrack
from metricas import evaluar
from visualizar import generar_video_tracking


VIDEO_LIST = [
    "uav0000117_02622_v",
    "uav0000137_00458_v",
    "uav0000182_00000_v",
    "uav0000268_05773_v",
    "uav0000305_00000_v",
    "uav0000339_00001_v",
]

METHODS = [
    {"name": "metodo1_sort",      "function": procesar_secuencia_sort},
    {"name": "metodo2_bytetrack", "function": procesar_secuencia_bytetrack},
    {"name": "metodo4_sahi",      "function": procesar_secuencia_sahi_bytetrack},
]

# Videos que quieres mostrar en la presentación para el método 4.
# 0182 y 0268 usan SAHI.
# 0305 es un caso difícil donde el análisis mostró que era mejor NO usar SAHI.
VIDEOS_DEMO_METODO4 = [
    "uav0000182_00000_v",   # mejor caso con SAHI
    "uav0000305_00000_v",   # caso difícil: YOLO low-conf sin SAHI
    "uav0000268_05773_v",   # caso difícil con SAHI en alta resolución
]


if __name__ == "__main__":
    # 0 = metodo1_sort
    # 1 = metodo2_bytetrack
    # 2 = metodo4_sahi
    metodo = 2

    nombre_metodo = METHODS[metodo]["name"]
    funcion_metodo = METHODS[metodo]["function"]

    resultados = []

    for video in VIDEO_LIST:
        secuencia = f"data/{video}"
        txt = f"outputs/{nombre_metodo}/{video}.txt"
        mp4 = f"outputs/{nombre_metodo}/{video}_resultado.mp4"
        gt = f"data/{video}/gt_mot_challenge/gt.txt"

        n_frames = len(glob.glob(os.path.join(secuencia, "img1", "*.jpg")))

        t0 = time.perf_counter()
        funcion_metodo(secuencia, txt)
        tiempo = time.perf_counter() - t0

        print(f"⏱️  Tiempo total: {tiempo:.2f}s  |  Frames: {n_frames}")

        # Generar videos solo para los casos que vas a mostrar en la presentación.
        if nombre_metodo == "metodo4_sahi" and video in VIDEOS_DEMO_METODO4:
            print(f"🎬 Generando video demo para presentación: {video}")
            generar_video_tracking(secuencia, txt, mp4)
            print(f"✅ Video generado en: {mp4}")

        metricas = evaluar(gt, txt, tiempo, n_frames)
        resultados.append({"video": video, **metricas})

    print(f"\n{'═' * 64}")
    print(f"  {'Video':<28}  {'MOTA':>6}  {'IDF1':>6}  {'ID-SW':>5}  {'FPS':>6}")
    print(f"{'─' * 64}")

    for r in resultados:
        fps_str = f"{r['fps']:.1f}" if r["fps"] is not None else "—"
        print(
            f"  {r['video']:<28}  "
            f"{r['mota'] * 100:>5.1f}%  "
            f"{r['idf1'] * 100:>5.1f}%  "
            f"{r['id_switches']:>5}  "
            f"{fps_str:>6}"
        )

    print(f"{'─' * 64}")

    mota_mean = sum(r["mota"] for r in resultados) / len(resultados)
    idf1_mean = sum(r["idf1"] for r in resultados) / len(resultados)
    ids_mean = sum(r["id_switches"] for r in resultados) / len(resultados)

    fps_values = [r["fps"] for r in resultados if r["fps"] is not None]
    fps_mean = sum(fps_values) / len(fps_values) if fps_values else None
    fps_str = f"{fps_mean:.1f}" if fps_mean is not None else "—"

    print(
        f"  {'MEAN':<28}  "
        f"{mota_mean * 100:>5.1f}%  "
        f"{idf1_mean * 100:>5.1f}%  "
        f"{ids_mean:>5.1f}  "
        f"{fps_str:>6}"
    )

    print(f"{'═' * 64}\n")