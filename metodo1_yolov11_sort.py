import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

def calcular_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

def procesar_secuencia_sort(path_secuencia, path_salida_txt):
    print(f"\n🚀 Procesando secuencia: {os.path.basename(path_secuencia)}")
    
    # Cargar YOLOv11 nano estándar
    model = YOLO("yolo11n.pt") 
    tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
    
    carpeta_img1 = os.path.join(path_secuencia, "img1")
    imagenes = sorted(glob.glob(os.path.join(carpeta_img1, "*.jpg")))
    
    if not imagenes:
        print("❌ No se encontraron imágenes.")
        return

    lineas_resultado_mot = []
    id_a_clase = {}

    for idx, path_img in enumerate(imagenes):
        frame_id = idx + 1
        img = cv2.imread(path_img)
        
        # Ejecutar inferencia con mayor resolución para objetos pequeños.
        predicciones = model(img, conf=0.35, imgsz=1024, verbose=False)[0]
        
        detecciones_para_tracker = []
        
        for box in predicciones.boxes:
            clase_coco = int(box.cls[0].item())
            confianza = float(box.conf[0].item())
            
            # MAPEO DE CLASES COCO -> VISDRONE:
            # En COCO: 0 es persona, 2 es carro.
            # En VisDrone (Guía): 1 es peatón, 4 es carro.
            clase_visdrone = None
            if clase_coco == 0:    # Persona
                clase_visdrone = 1
            elif clase_coco == 2:  # Carro
                clase_visdrone = 4
                
            # Si es una de las dos clases que nos interesan, la guardamos
            if clase_visdrone in [1, 4]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Pasamos también la clase al final para que no se pierda
                detecciones_para_tracker.append([x1, y1, x2, y2, confianza, clase_visdrone])
        
        if len(detecciones_para_tracker) == 0:
            tracks_actualizados = tracker.update(np.empty((0, 5)))
        else:
            # Para el tracker SORT estándar pasamos solo las primeras 5 columnas [x1, y1, x2, y2, conf]
            array_tracker = np.array([d[:5] for d in list(detecciones_para_tracker)])
            tracks_actualizados = tracker.update(array_tracker)
        
        # Guardar resultados
        for trk in tracks_actualizados:
            x1, y1, x2, y2, obj_id = trk
            obj_id = int(obj_id)
            bb_left = max(0, x1)
            bb_top = max(0, y1)
            bb_width = max(1, x2 - x1)
            bb_height = max(1, y2 - y1)

            clase_mot = id_a_clase.get(obj_id, 4)
            mejor_iou = 0.0
            for det in detecciones_para_tracker:
                iou = calcular_iou_xyxy([x1, y1, x2, y2], det[:4])
                if iou > mejor_iou:
                    mejor_iou = iou
                    clase_mot = int(det[5])

            if mejor_iou > 0:
                id_a_clase[obj_id] = clase_mot

            linea_mot = f"{frame_id},{obj_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},1.0,{clase_mot},1\n"
            lineas_resultado_mot.append(linea_mot)

    os.makedirs(os.path.dirname(path_salida_txt), exist_ok=True)
    with open(path_salida_txt, "w") as archivo_txt:
        archivo_txt.writelines(lineas_resultado_mot)
    print(f"✅ Archivo de texto actualizado correctamente.")
