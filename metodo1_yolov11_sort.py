import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

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

    for idx, path_img in enumerate(imagenes):
        frame_id = idx + 1
        img = cv2.imread(path_img)
        
        # Ejecutar inferencia con un umbral de confianza accesible (conf=0.25)
        predicciones = model(img, conf=0.25, verbose=False)[0]
        
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
            bb_left = max(0, x1)
            bb_top = max(0, y1)
            bb_width = max(1, x2 - x1)
            bb_height = max(1, y2 - y1)
            
            # Recuperar la clase real comparando la posición con nuestras detecciones originales
            clase_mot = 4  # Por defecto carro
            if bb_width < bb_height:
                clase_mot = 1  # Aproximación por aspecto si es más alto que ancho (Peatón)
            
            linea_mot = f"{frame_id},{int(obj_id)},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},1.0,{clase_mot},1\n"
            lineas_resultado_mot.append(linea_mot)

    os.makedirs(os.path.dirname(path_salida_txt), exist_ok=True)
    with open(path_salida_txt, "w") as archivo_txt:
        archivo_txt.writelines(lineas_resultado_mot)
    print(f"✅ Archivo de texto actualizado correctamente.")