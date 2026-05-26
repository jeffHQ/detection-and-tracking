from metodo1_yolov11_sort import procesar_secuencia_sort as metodo1
from metodo2_yolo26_bytetrack import procesar_secuencia_bytetrack as metodo2
from metricas import evaluar, procesar_y_medir
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
    {"name": "metodo1_sort", "function": metodo1},
    {"name": "metodo2_bytetrack", "function": metodo2}
]

if __name__ == "__main__":
    video = VIDEO_LIST[1]
    metodo = 1

    secuencia = f"data/{video}"
    txt = f"outputs/{METHODS[metodo]['name']}/{video}.txt"
    mp4 = f"outputs/{METHODS[metodo]['name']}/{video}_resultado.mp4"
    gt = f"data/{video}/gt_mot_challenge/gt.txt"

    # Ejecutar método y generar video
    METHODS[metodo]['function'](secuencia, txt)
    generar_video_tracking(secuencia, txt, mp4)

    # Evaluar
    tiempo, n_frames = procesar_y_medir(METHODS[metodo]['function'], secuencia, txt)
    evaluar(gt, txt, tiempo, n_frames)