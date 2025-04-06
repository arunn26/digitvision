# src/utils.py

import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plots training and validation loss/accuracy metrics."""
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Loss
    axs[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axs[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot Accuracy
    axs[1].plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    axs[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    axs[1].set_title('Accuracy (%)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    return fig
