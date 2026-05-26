# Laboratorio IV: Detección y Seguimiento de Objetos en Imágenes Aéreas (VisDrone)

Este repositorio contiene la implementación del laboratorio de **Multi-Object Tracking (MOT)** para la asignatura de Visión por Computador. El objetivo principal es evaluar el rendimiento de cuatro pipelines desacoplados de detección y asociación de datos en secuencias de video capturadas por drones (UAVs), enfocándose en las clases **Peatón (Clase 1)** y **Carro (Clase 4)** utilizando el dataset **VisDrone2019-MOT-val**.

---

## 👥 Integrantes / Autores
* [Ayuda me tienen encerrado en un calabozo desde hace años](https://github.com/tu_usuario) - 202020080

* [Luciano Aguirre Jesfen](https://github.com/lajesfen) - 202220401

* [Jeffry Arturo Hilario Quintana](https://github.com/jeffHQ) - 202020082


---

## 🛠️ Estructura del Proyecto

El proyecto está diseñado de forma modular, **desacoplando estrictamente** la etapa de inferencia del detector de la etapa de asociación geométrica/apariencia de los trackers para cumplir con las directrices de la guía de laboratorio.

```text
├── data/                       # Dataset VisDrone (Ignorado en Git)
│   ├── uav0000117_02622_v/
│   └── ...
├── outputs/                    # Archivos .txt de salida en formato MOT (Ignorado en Git)
│   ├── metodo1_sort/
│   ├── metodo2_bytetrack/
│   ├── metodo3_botsort/
│   └── metodo4_sahi/
├── sort.py                     # Implementación del algoritmo SORT clásico
├── metodo1_yolov11_sort.py     # Script ejecutable del Método Baseline (Masivo)
├── visualizar_metodo1.py       # Script para renderizar y exportar videos con bounding boxes
├── requirements.txt            # Dependencias del proyecto
└── .gitignore                  # Filtros de archivos pesados (.pt, data/, outputs/)