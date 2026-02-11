# Use a lightweight Python image with PyTorch pre-installed
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts from the src folder to the container's /app folder
COPY src/ /app/

# We don't define a CMD because Kubeflow will override it 
# to run preprocess.py, train.py, etc., individually.
