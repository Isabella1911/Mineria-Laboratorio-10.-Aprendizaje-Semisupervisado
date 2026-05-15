# Laboratorio 10 - Aprendizaje Semisupervisado

Este repositorio contiene una entrega reproducible para el laboratorio de
aprendizaje semisupervisado de Mineria de Datos.

## Dataset

Se utiliza el dataset publico **Wine Quality - Red Wine** de UCI:

https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

Cumple los requisitos del laboratorio: es real, publico, tiene 1599 filas y 12
columnas (11 variables predictoras numericas y la variable objetivo `quality`).

## Como ejecutar

```powershell
python lab_semisupervisado.py
```

El script no requiere `pandas`, `scikit-learn` ni `matplotlib`; implementa el
flujo con `numpy` y librerias estandar de Python para que pueda ejecutarse en
este entorno.

## Material generado

Al ejecutar el script se crean:

- `data/winequality-red.csv`: dataset descargado.
- `outputs/results.csv`: resultados detallados de todos los experimentos.
- `outputs/summary.txt`: resumen numerico de la ejecucion.
- `outputs/reporte_laboratorio_10.pdf`: reporte tecnico en PDF.
- `outputs/figures/*.svg`: graficos usados para analizar los resultados.

## Modelos comparados

- Baseline supervisado: KNN entrenado solo con el subconjunto etiquetado.
- Semisupervisado 1: Self-Training con KNN y pseudo-etiquetado por umbral.
- Semisupervisado 2: Label Propagation sobre grafo k-NN.

Los experimentos simulan 5%, 10% y 20% de datos etiquetados, usando el resto de
los datos de entrenamiento como no etiquetados.
