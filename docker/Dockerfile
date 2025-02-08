FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y \
        libgl1-mesa-glx \    
        libglib2.0-0

COPY ./sources/ /workspace

WORKDIR /workspace

RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir -r requirements.txt

USER 1000:1000

CMD ["python", "model-builder.py", "--help"]

# TODO: Add runner stage
