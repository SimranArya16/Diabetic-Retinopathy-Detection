# Diabetic Retinopathy Detection using Deep Learning & Explainable AI

## Overview
This project is an AI-based system that detects and classifies Diabetic Retinopathy (DR) from retinal fundus images.  
It uses deep learning along with Explainable AI techniques like Grad-CAM and LIME to provide visual explanations for predictions.  
The system also generates a clinical-style PDF report for better interpretation.

## Features
- Classification of DR stages (No DR, Mild, Moderate, Severe, Proliferative)
- Grad-CAM++ heatmap visualization
- LIME-based explanation
- Automated PDF report generation
- Streamlit-based web interface

## Model Details
The model is based on EfficientNet (pretrained) for feature extraction.  
It is trained using labeled fundus images with 5 DR classes.  
Evaluation metrics include accuracy, precision, recall, and F1-score.

## How to Run

### Install dependencies
