FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch

WORKDIR /workspace

CMD ["bash"]

# Start with
# podman run -it -v "$(pwd)":/workspace llm_pod
