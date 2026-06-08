import os
import shutil

from visualizar import generar_video_tracking


VIDEOS_SAHI = [
    "uav0000182_00000_v",
    "uav0000268_05773_v",
]

METODO = "metodo4_sahi"
OUTPUT_TXT_DIR = f"outputs/{METODO}"
OUTPUT_VIDEO_DIR = "outputs/videos_finales_sahi"


if __name__ == "__main__":
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    print("🎬 Generando videos finales SOLO para los videos que usan SAHI")
    print(f"📁 Carpeta de salida: {OUTPUT_VIDEO_DIR}")

    for video in VIDEOS_SAHI:
        secuencia = f"data/{video}"
        txt_origen = f"{OUTPUT_TXT_DIR}/{video}.txt"
        mp4_salida = f"{OUTPUT_VIDEO_DIR}/{video}_SAHI_FINAL.mp4"
        txt_copia = f"{OUTPUT_VIDEO_DIR}/{video}_SAHI_FINAL.txt"

        print("\n" + "=" * 80)
        print(f"Video: {video}")
        print(f"Secuencia: {secuencia}")
        print(f"TXT origen: {txt_origen}")
        print(f"MP4 salida: {mp4_salida}")

        if not os.path.exists(secuencia):
            print(f"❌ No existe la secuencia: {secuencia}")
            continue

        if not os.path.exists(txt_origen):
            print(f"❌ No existe el archivo TXT: {txt_origen}")
            print("   Primero corre main.py para generar los resultados del método 4.")
            continue

        # Copia también el txt usado para que quede evidencia exacta
        # de qué tracking se visualizó en el video.
        shutil.copy(txt_origen, txt_copia)

        generar_video_tracking(
            secuencia,
            txt_origen,
            mp4_salida,
        )

        print(f"✅ Video generado: {mp4_salida}")
        print(f"✅ TXT copiado: {txt_copia}")

    print("\n✅ Listo. Videos finales generados en:")
    print(f"   {OUTPUT_VIDEO_DIR}")