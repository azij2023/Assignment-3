# Use a lightweight Python image
FROM python:3.13-slim

# Update system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y build-essential && apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get remove --purge -y build-essential && apt-get autoremove -y

# Set working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install --upgrade -r requirements.txt

# Run your prediction script
CMD ["python", "src/predict.py"]
