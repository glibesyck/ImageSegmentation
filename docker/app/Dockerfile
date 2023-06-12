FROM python:3.9-slim

COPY ["../../app/main.py", "../../app/predict.py", "../../app/reduced_model.h5", "/opt/app/"]
COPY ["../../app/templates/index.html", "/opt/app/templates/"]
COPY ["../../config/app_requirements.txt", "/opt/app/"]

WORKDIR /opt/app

ENV PYTHONPATH "${PYTHONPATH}:/"

RUN pip install --no-cache-dir -r app_requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]