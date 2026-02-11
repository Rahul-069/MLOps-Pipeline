FROM python:3.9-slim
RUN pip install torch torchvision
WORKDIR /app
COPY src/ /app/
# Ensure the scripts are in the root of /app