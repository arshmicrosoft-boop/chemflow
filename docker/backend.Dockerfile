FROM python:3.11-slim
WORKDIR /workspaces/project
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/app ./backend/app
COPY chemflow ./chemflow
COPY datasets ./datasets
ENV PYTHONPATH=/workspaces/project
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
