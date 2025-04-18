# Use a lightweight base image
FROM python:3.10-slim

# Set env variables to prevent Python from writing .pyc files and to ensure output is sent straight to the terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your project
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
