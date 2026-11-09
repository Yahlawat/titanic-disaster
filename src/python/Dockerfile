# Step 12: Dockerfile that installs requirements.txt and runs the .py script
FROM python:3.13.13-slim

WORKDIR /app

# install packages first so code changes don't redo this step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic.py .

CMD ["python", "titanic.py"]
