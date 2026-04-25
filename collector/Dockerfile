FROM python:3.12-slim

WORKDIR /app

COPY lgmonitor /lgmonitor
RUN pip install --no-cache-dir /lgmonitor

COPY collector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY collector/ .

CMD ["python", "-m", "dummy_graph.runner"]
