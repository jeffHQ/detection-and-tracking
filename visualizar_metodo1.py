import os
import glob
import cv2

def generar_video_tracking(path_secuencia, path_txt_resultados, video_salida_mp4):
    print(f"🎬 Generando video para la secuencia: {os.path.basename(path_secuencia)}")
    
    # 1. Cargar las anotaciones generadas por tu Método 1 en un diccionario
    # Estructura: { frame_id: [[id, x, y, w, h, clase], ...] }
    tracking_por_frame = {}
    if not os.path.exists(path_txt_resultados):
        print(f"❌ Error: No se encontró el archivo de texto en {path_txt_resultados}")
        return
        
    with open(path_txt_resultados, "r") as f:
        for linea in f:
            partes = linea.strip().split(',')
            if len(partes) < 8:
                continue
            frame = int(partes[0])
            obj_id = int(partes[1])
            x = float(partes[2])
            y = float(partes[3])
            w = float(partes[4])
            h = float(partes[5])
            clase = int(partes[7])
            
            if frame not in tracking_por_frame:
                tracking_por_frame[frame] = []
            tracking_por_frame[frame].append([obj_id, x, y, w, h, clase])

    # 2. Obtener las imágenes originales en orden numérico estricto
    carpeta_img1 = os.path.join(path_secuencia, "img1")
    imagenes = sorted(glob.glob(os.path.join(carpeta_img1, "*.jpg")))
    
    if not imagenes:
        print("❌ Error: Carpeta de imágenes vacía.")
        return

    # 3. Leer la primera imagen para configurar las dimensiones del video de salida
    img_muestra = cv2.imread(imagenes[0])
    alto, ancho, _ = img_muestra.shape
    
    # Configurar el escritor de video OpenCV (usando el códec mp4v que es estándar)
    os.makedirs(os.path.dirname(video_salida_mp4), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_salida_mp4, fourcc, 20.0, (ancho, alto))

    # 4. Recorrer fotograma por fotograma pintando las cajas almacenadas
    for idx, path_img in enumerate(imagenes):
        frame_id = idx + 1
        img = cv2.imread(path_img)
        
        # Si el frame actual tiene registros en nuestro archivo .txt, los dibujamos
        if frame_id in tracking_por_frame:
            for obj_id, x, y, w, h, clase in tracking_por_frame[frame_id]:
                # Definir color según la clase (Verde para Carro, Azul para Peatón)
                color = (0, 255, 0) if clase == 4 else (255, 0, 0)
                label = f"ID {obj_id}: {'Car' if clase == 4 else 'Pedestrian'}"
                
                # Coordenadas enteras para OpenCV
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # Dibujar la caja delimitadora
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Dibujar fondo del texto para mejorar la lectura
                cv2.rectangle(img, (x1, y1 - 18), (x1 + 130, y1), color, -1)
                cv2.putText(img, label, (x1 + 3, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Escribir el fotograma procesado en el archivo de video masivo
        video_writer.write(img)
        
    video_writer.release()
    print(f"✅ ¡Video guardado con éxito en! 🎥 {video_salida_mp4}")

# --- Bloque ejecutor ---
if __name__ == "__main__":
    secuencia = "data/uav0000117_02622_v"
    txt_origen = "outputs/metodo1_sort/uav0000117_02622_v.txt"
    video_destino = "outputs/metodo1_sort/uav0000117_02622_v_resultado.mp4"
    
    generar_video_tracking(secuencia, txt_origen, video_destino)