<a href="https://digitvision-aasheik.streamlit.app/" target="_blank">🚀 <strong>Feeling Lazy? Just Click Here for the Live Demo!</strong> 🚀</a>

# DigitVision: Analog Neural Network Explorer

Welcome, lazy genius! If setting things up manually feels like a drag, click the live demo link above and dive straight into DigitVision. For the brave souls (or the simply curious) who want to poke around under the hood, read on!

## Overview

DigitVision is an interactive playground for a quirky analog neural network built to recognize handwritten digits (MNIST dataset). Here's what you get:

- **Interactive Visualization:** Peek inside an analog neural network with a user-friendly UI.
- **Model Training & Evaluation:** Load a trained model or retrain with your own hyperparameters.
- **Metrics Visualization:** See cool graphs of loss and accuracy over training epochs.
- **Sample Predictions:** Let the network show off by predicting random MNIST digits.
- **Future Enhancements:** Soon, you'll be doodling digits live and watching predictions happen!

## Repository Structure

```
├── frontend/
│   └── app.py            # Streamlit UI magic happens here
├── src/
│   ├── ann_model.py      # Our analog neural network brains
│   ├── data_loader.py    # Fetches MNIST goodies
│   ├── train.py          # Train the model like a command-line pro
│   ├── utils.py          # Handy functions for metrics plotting
│   └── __init__.py       # Python package essentials
└── README.md             # Yep, this fun little file
```

## Installation

If you're the adventurous DIY type:

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/digitvision.git
cd digitvision
```

### 2. Virtual Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows folks: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Frontend Fun

Launch the shiny Streamlit app:

```bash
streamlit run frontend/app.py
```

- **Sidebar Goodies:**
  - **Load Trained Model:** Skip training and load saved weights.
  - **Retrain Model:** Customize your network's hidden layers, epochs, and learning rate.

- **Explore Tabs:**
  - **Home:** Quick intro & sample MNIST digits.
  - **Trained Metrics:** Visualize training metrics and peek at model weights.
  - **Sample Prediction:** Model predicts random MNIST samples.
  - **Draw Digit (Coming Soon):** Your doodles, live predictions!

### Command-line Warrior

Run this script to train and evaluate the model directly:

```bash
python src/train.py
```

It will:
- Grab MNIST data
- Train the network
- Evaluate accuracy
- Save the model
- Plot your epic training journey

## Customization

Tweak hyperparameters easily from the sidebar:
- Hidden layer size
- Epochs
- Learning rate

Weights and metrics (`model_weights.npz`, `training_metrics.npz`) auto-save for easy reloading.

## Dependencies

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [Pillow](https://python-pillow.org/)

## Coming Soon

- **Digit Drawing:** Scribble digits and get real-time predictions.
- **Advanced Models:** Play around with deeper and smarter architectures.
- **Speed Boosts:** Performance improvements for faster inference.

## Acknowledgments

- MNIST dataset by [Yann LeCun](http://yann.lecun.com/exdb/mnist/).
- Inspired by analog neural networks and playful ML visualizations.
