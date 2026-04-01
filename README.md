# User Directory & Prediction API

A full-stack application combining a React frontend and Flask backend to manage user data and perform house price predictions using a machine learning model.

## Features
- RESTful API with CRUD operations (GET, POST, PUT, DELETE)
- React frontend for dynamic user interaction
- House price prediction using a trained machine learning model
- JSON data processing and validation

## Tech Stack
- React
- Flask (Python)
- JavaScript
- HTML/CSS

## Overview
This project demonstrates full-stack development by integrating a frontend interface with backend APIs and handling real-time data exchange between systems.

## Note
This repository does not include the trained model file (`random_forest_model.pkl`). The application structure, API logic, and frontend/backend integration are included for demonstration purposes.

## How to Run

### Frontend
cd frontend
npm install
npm start

### Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
