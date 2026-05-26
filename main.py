from metodo1_yolov11_sort import procesar_secuencia_sort as metodo1
from metodo2_yolo26_bytetrack import procesar_secuencia_bytetrack as metodo2
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
    "metodo1_sort",
    "metodo2_bytetrack"
]

if __name__ == "__main__":
    video = VIDEO_LIST[1]
    metodo = METHODS[1]
    secuencia = f"data/{video}"
    txt = f"outputs/{metodo}/{video}.txt"
    mp4 = f"outputs/{metodo}/{video}_resultado.mp4"

    metodo2(secuencia, txt)
    generar_video_tracking(secuencia, txt, mp4)