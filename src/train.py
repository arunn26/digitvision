# File: src/train.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.data_loader import get_data_loaders
from src.ann_model import ImprovedAnalogNeuralNetwork
from src.utils import plot_metrics
from src.config import MODEL_PATH, METRICS_PATH, DEFAULT_HIDDEN_SIZE, DEFAULT_EPOCHS, DEFAULT_LR

def main():
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

    # Initialize model
    ann = ImprovedAnalogNeuralNetwork(input_size=784, hidden_size=DEFAULT_HIDDEN_SIZE, output_size=10)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = ann.train(
        train_loader, val_loader,
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LR
    )

    # Evaluate on test set
    test_accuracy = ann.evaluate(test_loader)

    # Save the trained model
    if not os.path.exists("models"):
        os.makedirs("models")
    ann.save(MODEL_PATH)

    # Save training metrics
    np.savez(
        METRICS_PATH,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        train_accuracies=np.array(train_accuracies),
        val_accuracies=np.array(val_accuracies),
        test_accuracy=test_accuracy
    )

    # Optionally plot metrics
    fig = plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    fig.show()

if __name__ == '__main__':
    main()
