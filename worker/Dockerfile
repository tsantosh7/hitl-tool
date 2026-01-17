FROM python:3.11-slim
WORKDIR /worker
COPY requirements.txt /worker/
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py /worker/worker.py
CMD ["python", "worker.py"]
