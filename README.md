# 🧠 DigitVision - Analog Neural Network Digit Recognition

DigitVision is a Streamlit-based web application that lets you draw digits (0–9) and recognizes them using a custom-built analog neural network. 

---

## 📸 Features

- 🎨 **Interactive Canvas:** Draw digits directly on your browser.
- ⚡ **Real-Time Prediction:** Instantly recognizes digits using an analog neural network.
- 📈 **Visualization:** View detailed training & validation metrics.
- 🔁 **Retraining:** Train the model within the app using customizable hyperparameters.
- 💾 **Model Persistence:** Save & Load trained model weights and training logs.

---

## 🚀 Live Demo

> _Coming soon: Deployment on [Streamlit Cloud](https://streamlit.io/cloud)
---

## 📁 Project Structure

```
digitvision/
├── frontend/
│   ├── app.py                # Main Streamlit application
│   └── draw_canvas.py        # Canvas drawing and preprocessing utilities
├── src/
│   ├── ann_model.py          # Analog neural network class
│   ├── data_loader.py        # Data loader and preprocessing
│   ├── utils.py              # Utility functions for plotting
│   ├── train.py              # Training script for CLI
│   └── config.py             # Central configuration parameters
├── models/                   # Directory for model weights & logs
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation (you are here)
```

---

## ⚙️ Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/digitvision.git
cd digitvision
```

### 2. Set up Python Environment

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** Requires Python **3.7 or higher**.

## 📚 Dependencies

```
streamlit
streamlit-drawable-canvas
numpy
matplotlib
Pillow
torch
torchvision
```

---

## 🎯 Running the App

```bash
streamlit run frontend/app.py
```

## 🧪 Retraining the Neural Network

1. Open sidebar options and tick ✅ **"Retrain Model"**.
2. Configure desired hyperparameters (epochs, learning rate, hidden layer size).
3. Click the **"Start Retraining"** button to initiate.
4. After training, weights and logs save automatically in `models/`.

---