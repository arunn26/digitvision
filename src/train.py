import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.data_loader import get_data_loaders
from src.ann_model import ImprovedAnalogNeuralNetwork
from src.utils import plot_metrics

def main():
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

    # Initialize model with default hyperparameters
    ann = ImprovedAnalogNeuralNetwork(input_size=784, hidden_size=100, output_size=10)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = ann.train(
        train_loader, val_loader, epochs=10, learning_rate=0.001
    )

    # Evaluate on test set
    ann.evaluate(test_loader)

    # Save the trained model
    if not os.path.exists("models"):
        os.makedirs("models")
    ann.save("models/model_weights.npz")

    # Save training metrics
    import numpy as np
    test_accuracy = ann.evaluate(test_loader)
    np.savez("models/training_metrics.npz",
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses),
             train_accuracies=np.array(train_accuracies),
             val_accuracies=np.array(val_accuracies),
             test_accuracy=test_accuracy)

    # Plot metrics
    fig = plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    fig.show()

if __name__ == '__main__':
    main()
