import torch
from src.data_loader import get_data_loaders
from src.ann_model import ImprovedAnalogNeuralNetwork
from src.utils import plot_metrics

def main():
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Initialize model with default hyperparameters
    ann = ImprovedAnalogNeuralNetwork(input_size=784, hidden_size=100, output_size=10)
    
    # Train the model (adjust epochs and learning_rate as needed)
    train_losses, val_losses, train_accuracies, val_accuracies = ann.train(train_loader, val_loader, epochs=10, learning_rate=0.001)
    
    # Evaluate on test set
    ann.evaluate(test_loader)
    
    # Save the trained model
    ann.save("model_weights.npz")
    
    # Plot and display metrics
    fig = plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    fig.show()

if __name__ == '__main__':
    main()
