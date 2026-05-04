"""
Aplicación FastAPI ... usando un modelo de Machine Learning.

Esta API permite hacer predicciones individuales y por lotes sobre si .... 
Está diseñada con fines didácticos para enseñar el uso de FastAPI en aplicaciones de ML.

Características principales:
- Endpoint /predict: Predicción individual de retraso.
- Endpoint /predict-batch: Predicciones por lotes.
- Endpoint /info: Información sobre la API.
- Endpoint /metrics: Métricas de uso (número de predicciones, tiempo de actividad).
- Endpoint /simulate-error: Simulación de errores para testing.

El modelo utilizado es un ... entrenado previamente y guardado en '...pkl'.
"""

from pydantic import BaseModel, Field
import joblib
from fastapi import FastAPI, HTTPException
import time
import numpy as np

# ------------------------- Modelo de datos Pydantic -------------------------
# Definimos un modelo Pydantic para validar automáticamente las entradas.
# Esto permite que FastAPI genere documentación interactiva en /docs con campos editables.
class FlowerFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=1000, description="Longitud del sépalo en cm")
    sepal_width: float = Field(..., ge=0, le=1000, description="Ancho del sépalo en cm")
    petal_length: float = Field(..., ge=0, le=1000, description="Longitud del pétalo en cm")
    petal_width: float = Field(..., ge=0, le=1000, description="Ancho del pétalo en cm")

CLASSES = ["setosa", "versicolor", "virginica"]

# ------------------------- Cargar modelo entrenado -------------------------
# El modelo de Machine Learning se carga al iniciar la aplicación.
# Se utiliza joblib para deserializar el modelo guardado en un archivo pickle.
# Si no se puede cargar, se lanza un error crítico que detiene la aplicación.
try: 
    model = joblib.load('./model_RFC.pkl')
    print(f"Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise RuntimeError(f"No se pudo cargar el modelo de Machine Learning. Verifique el archivo {model}.")



# ------------------------- Inicializar FastAPI -------------------------
app = FastAPI(title="API de Predicción de Flores Iris", description="API para hacer predicciones de iris usando Machine Learning.")


# Contadores y estado simple para métricas
# prediction_count: Número total de predicciones realizadas desde el inicio.
# start_time: Momento en que se inició la aplicación, para calcular el uptime.
prediction_count = 0
start_time = time.time()



# ------------------------- Endpoint de predicción individual -------------------------
# Endpoint POST para hacer una predicción individual
#
# Método: POST
# Ruta: /predict
# Cuerpo de la solicitud: JSON con
#
# Proceso:
#
# Respuesta exitosa (200):
#{
#    "sepal_length": 5.1,
#    "sepal_width": 3.5,
#    "petal_length": 1.4,
#    "petal_width": 0.2
#}
# Errores posibles:
# - 422: Datos inválidos (Pydantic validation error).
# - 500: Error interno (problema con el modelo o procesamiento).
@app.post("/predict-batch")
async def predict_batch(features_list: list[FlowerFeatures]):
    try:
        results = []
        for features in features_list:
            # Convertir entrada a array 2D
            input_data = [[
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ]]

            # Obtener probabilidades
            probabilities = model.predict_proba(input_data)[0]

            # Índice y clase predicha
            index = int(np.argmax(probabilities))
            prediction = CLASSES[index]
            confidence = float(probabilities[index])

            # Construir respuesta
            result = {
                "prediction": prediction,
                "prediction_index": index,
                "probabilities": {
                    "setosa": float(probabilities[0]),
                    "versicolor": float(probabilities[1]),
                    "virginica": float(probabilities[2])
                },
                "confidence": confidence,
                "status": "ok"
            }
            results.append(result)

        return {"predictions": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")



# ------------------------- Endpoint predict-batch -------------------------
#
# Método: POST
# Ruta: /predict-batch
# Cuerpo de la solicitud: Lista de JSON, cada uno con
#
# Proceso:
#
# Respuesta exitosa (200):
# {
#   "predictions": [
#     {
#       "prediction": "setosa",
#       "prediction_index": 0,
#       "probabilities": {
#        "setosa": 0.97,
#        "versicolor": 0.02,
#        "virginica": 0.01
#       },
#       "confidence": 0.97,
#       "status": "success"
#      },
#     ...
#   ],
#   "count": 5
# }
#
# Notas:
# - inválidos se ignoran silenciosamente (no se incluyen en resultados).
# - Útil para procesar múltiples vuelos en una sola solicitud, optimizando rendimiento.
@app.post("/predict")
async def predict_batch(features: FlowerFeatures):
    global prediction_count
    try:
        input_data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]]
        # Obtener probabilidades
        probabilities = model.predict_proba(input_data)[0]

        # Índice y clase predicha
        index = int(np.argmax(probabilities))
        prediction = CLASSES[index]
        confidence = float(probabilities[index])
        prediction_count += 1

        return {
            "prediction": prediction,
            "prediction_index": index,
            "probabilities": {
                "setosa": float(probabilities[0]),
                "versicolor": float(probabilities[1]),
                "virginica": float(probabilities[2])
            },
            "confidence": confidence,
            "status": "ok"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ------------------------- Endpoint info -------------------------
# Endpoint GET para obtener información general sobre la API.
#
# Método: GET
# Ruta: /info
# Sin parámetros.
#
# Retorna metadatos de la API, incluyendo nombre, descripción, versión y lista de características disponibles.
# Útil para que los clientes conozcan las capacidades del servicio.
#
# Respuesta (200):
@app.get("/")
def info():
    return {
        "name": "API de Predicción de Flores Iris",
        "description": "API para hacer predicciones de iris usando Machine Learning.",
        "version": "1.0.0",
        "features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    }



# ------------------------- Endpoint metrics -------------------------
# Endpoint GET para obtener métricas de uso de la API.
#
# Método: GET
# Ruta: /metrics
# Sin parámetros.
#
# Calcula y retorna:
# - total_predictions: Número total de predicciones realizadas desde el inicio.
# - uptime_seconds: Tiempo en segundos que la aplicación ha estado ejecutándose.
#
# Útil para monitoreo y debugging del servicio.
#
# Respuesta (200):
# {
#   "total_predictions": 42,
#   "uptime_seconds": 3600
# }
@app.get("/metrics")
def metrics():
    # Calcular uptime restando el tiempo actual al tiempo de inicio
    uptime = int(time.time() - start_time)
    return {
        "total_predictions": prediction_count,
        "uptime_seconds": uptime
    }

@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
