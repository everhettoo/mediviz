FROM --platform=linux/amd64 python:3.11-slim

# Install system dependencies required by PyQt5
RUN apt-get update && apt-get install -y \
    libgl1 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libx11-xcb1 \
    libxcb1 \
    libxrender1 \
    libxi6 \
    libxtst6 \
    libnss3 \
    && rm -rf /var/lib/apt/lists/*

# Fix Qt display issues
ENV QT_X11_NO_MITSHM=1

WORKDIR /mediviz

# Copy requirements first (better cache)
COPY requirements.txt .

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies (INCLUDING PyQt5 from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

CMD ["python", "app.py"]