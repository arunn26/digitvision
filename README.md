# DigitVision: Analog Neural Network Explorer

DigitVision is an interactive explorer for a custom analog neural network designed for handwritten digit recognition using the MNIST dataset. This repository contains both the front-end Streamlit application for visualization and interaction, as well as the backend training and model implementation.

## Overview

DigitVision provides:
- **Interactive Visualization:** Explore the inner workings of an analog neural network with a responsive UI.
- **Model Training & Evaluation:** Load a pre-trained model or retrain the network with custom hyperparameters.
- **Metrics Visualization:** Monitor training and validation loss/accuracy over epochs with detailed plots.
- **Sample Predictions:** Test the model on randomly selected MNIST samples.
- **Future Enhancements:** Features such as drawing a digit for real-time prediction are in development.

## Repository Structure

```
├── frontend/
│   └── app.py            # Streamlit application with the UI and interactivity
├── src/
│   ├── ann_model.py      # Implementation of the analog neural network and its training routines
│   ├── data_loader.py    # Data loader for MNIST dataset and helper functions to create data loaders
│   ├── train.py          # Script to train the model from the command line
│   ├── utils.py          # Utility functions (e.g., metrics plotting)
│   └── __init__.py       # Python package initializer
└── README.md             # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/digitvision.git
cd digitvision
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Ensure you have PyTorch, torchvision, Streamlit, NumPy, matplotlib, and Pillow installed. Adjust versions according to your system.

## Usage

### Running the Frontend

Launch the Streamlit interactive interface:

```bash
streamlit run frontend/app.py
```

- **Sidebar Options:**
  - **Load Trained Model:** Load pre-saved model weights.
  - **Retrain Model:** Retrain network with customized hyperparameters like hidden layer size, epochs, and learning rate.

- **Tabs:**
  - **Home:** Project overview and MNIST examples.
  - **Trained Metrics:** Visualize loss and accuracy plots, inspect model weights.
  - **Sample Prediction:** Run a random test sample prediction.
  - **Draw Digit (Coming Soon):** Placeholder for future feature.

### Training via Command Line

Alternatively, train your model directly:

```bash
python src/train.py
```

This script:
- Loads MNIST dataset
- Initializes and trains the neural network
- Evaluates performance on the test set
- Saves trained model weights
- Displays training metrics

## Customization

Adjust hyperparameters via Streamlit sidebar:
- Hidden layer size
- Epoch count
- Learning rate

Saved model weights (`model_weights.npz`) and training metrics (`training_metrics.npz`) can be loaded through the UI.

## Dependencies

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [Pillow](https://python-pillow.org/)

## Future Enhancements

- **Digit Drawing Feature:** Interactive canvas for real-time digit prediction.
- **Advanced Neural Network Architectures:** Experiment with deeper or alternative analog-inspired components.
- **Optimizations:** Enhance training and inference speed.

## Acknowledgments

- MNIST dataset from [Yann LeCun](http://yann.lecun.com/exdb/mnist/).
- Inspired by analog neural networks and interactive ML visualization.
