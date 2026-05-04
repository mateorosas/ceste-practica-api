FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos dependencias primero para aprovechar cache
COPY requirements.txt /app/requirements.txt

# Instalamos dependencias
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiamos el codigo
COPY . /app

# Puerto comun para APIs (ajusta si tu app usa otro)
EXPOSE 8080

# Comando por defecto (ajusta si usas uvicorn o flask)
# CMD ["python", "app.py"]
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:${PORT}"]
