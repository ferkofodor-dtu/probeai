# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install DVC with GCS support
RUN pip install 'dvc[gs]'

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY probeai/ probeai/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

#ENTRYPOINT ["python", "-u", "probeai/train.py"]
