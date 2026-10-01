FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY config ./config
RUN pip install --no-cache-dir -e '.[dashboard,borsapy]'
ENV PYTHONUNBUFFERED=1
# default: serve the dashboard; the scanner service overrides the command
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
