# Use the official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage caching
COPY requirements.txt .

# Install dependencies first (caching layer)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the full project (to avoid invalidating cache)
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
