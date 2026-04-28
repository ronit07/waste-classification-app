# Waste Classification App

A deep learning app that classifies photographed items into **recyclable**, **organic**, or **landfill** categories — taking the guesswork out of waste sorting.

## Overview

Fine-tuned **MobileNetV2** via transfer learning on ~2,500 labeled waste images. Applied data augmentation (flips, brightness adjustment, rotation) to maximize accuracy on a limited dataset.

## Model Performance

- **Architecture**: MobileNetV2 (transfer learning from ImageNet)
- **Dataset**: ~2,500 labeled images across 3 categories
- **Validation accuracy**: ~82%
- **Inference time**: < 2 seconds end-to-end

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML | Python, TensorFlow, Keras |
| Backend | Flask REST API |
| Frontend | React |

## Features

- Upload a photo of any item via the React frontend
- Receive a **disposal category** (recyclable / organic / landfill)
- See a **confidence score** alongside the prediction

## Project Structure

```
waste-classification-app/
├── model/
│   ├── train.py          # MobileNetV2 fine-tuning
│   ├── augment.py        # Data augmentation pipeline
│   └── model.h5          # Saved model weights
├── api/
│   └── app.py            # Flask REST endpoint
├── frontend/
│   └── src/              # React upload + result UI
└── data/
    └── README.md         # Dataset instructions
```

## Getting Started

```bash
# Train model
pip install -r requirements.txt
python model/train.py

# Run API
python api/app.py

# Run frontend
cd frontend && npm install && npm start
```
