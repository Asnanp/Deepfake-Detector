# Real-time Deepfake Detector

A web application that allows users to upload videos or images and get an instant authenticity score to detect deepfakes.

## Features

- Upload video or image files
- Real-time deepfake detection
- Authenticity score visualization
- Responsive design

## Project Structure

- `backend/`: Flask API for deepfake detection
- `frontend/`: HTML/CSS/JS for the user interface

## Setup and Installation

### Prerequisites

- Python 3.8+
- pip
- Node.js (optional, for frontend development)

### Installation

1. Clone the repository
2. Install backend dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and navigate to `http://localhost:5000`

## How It Works

The application uses deep learning models to analyze uploaded media and determine if it's authentic or a deepfake. The analysis results in an authenticity score that indicates the likelihood of the content being genuine.