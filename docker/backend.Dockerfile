FROM python:3.11-slim
WORKDIR /workspaces/project
# libxrender1 and libxext6 are needed by RDKit's drawing libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/app ./backend/app
COPY chemflow ./chemflow
COPY deepchem ./deepchem
COPY datasets/delaney-processed.csv datasets/crystal_desc_canvas_aug30.csv datasets/desc_canvas_aug30.csv datasets/tox21.csv.gz ./datasets/
ENV PYTHONPATH=/workspaces/project
ENV PYTHONUNBUFFERED=1
# Hosts such as Render choose the port through $PORT.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
