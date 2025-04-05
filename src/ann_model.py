import numpy as np

def pixels_to_voltages(pixels, min_voltage=-5.0, max_voltage=5.0):
    """Converts pixel values (normalized 0-1) to voltage range."""
    return pixels * (max_voltage - min_voltage) + min_voltage

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class ImprovedAnalogNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight initialization using uniform distribution based on layer sizes
        limit1 = np.sqrt(6 / (input_size + hidden_size))
        self.Gm1 = np.random.uniform(-limit1, limit1, (hidden_size, input_size))
        self.b1 = np.zeros(hidden_size)

        limit2 = np.sqrt(6 / (hidden_size + output_size))
        self.Gm2 = np.random.uniform(-limit2, limit2, (output_size, hidden_size))
        self.b2 = np.zeros(output_size)

    def forward(self, X_voltages):
        self.z1 = np.dot(self.Gm1, X_voltages) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.Gm2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X_voltages, y_onehot, output, learning_rate=0.001):
        error = output - y_onehot
        delta2 = error * sigmoid_derivative(self.z2)
        dGm2 = np.outer(delta2, self.a1)

        error_hidden = np.dot(self.Gm2.T, delta2)
        delta1 = error_hidden * relu_derivative(self.z1)
        dGm1 = np.outer(delta1, X_voltages)

        self.Gm2 -= learning_rate * dGm2
        self.b2 -= learning_rate * delta2
        self.Gm1 -= learning_rate * dGm1
        self.b1 -= learning_rate * delta1

    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001, log_callback=None):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            total_train_loss = 0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.view(-1, 28*28).numpy()
                labels_onehot = np.eye(self.output_size)[labels.numpy()]

                for X_pixels, y_onehot in zip(images, labels_onehot):
                    X_voltages = pixels_to_voltages(X_pixels)
                    output = self.forward(X_voltages)
                    loss = -np.sum(y_onehot * np.log(output + 1e-10))
                    total_train_loss += loss

                    self.backward(X_voltages, y_onehot, output, learning_rate=learning_rate)

                    if np.argmax(output) == np.argmax(y_onehot):
                        train_correct += 1
                    train_total += 1

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            train_accuracy = 100 * train_correct / train_total

            total_val_loss = 0
            val_correct = 0
            val_total = 0

            for images, labels in val_loader:
                images = images.view(-1, 28*28).numpy()
                labels_onehot = np.eye(self.output_size)[labels.numpy()]

                for X_pixels, y_onehot in zip(images, labels_onehot):
                    X_voltages = pixels_to_voltages(X_pixels)
                    output = self.forward(X_voltages)
                    loss = -np.sum(y_onehot * np.log(output + 1e-10))
                    total_val_loss += loss

                    if np.argmax(output) == np.argmax(y_onehot):
                        val_correct += 1
                    val_total += 1

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            val_accuracy = 100 * val_correct / val_total

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            log_msg = (f"Epoch {epoch+1}/{epochs}, "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            print(log_msg)
            if log_callback:
                log_callback(log_msg)

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X_voltages):
        output = self.forward(X_voltages)
        return np.argmax(output)

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28*28).numpy()
            labels = labels.numpy()
            for X_pixels, label in zip(images, labels):
                X_voltages = pixels_to_voltages(X_pixels)
                if self.predict(X_voltages) == label:
                    correct += 1
                total += 1
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def save(self, file_path):
        """Saves model weights to a file."""
        np.savez(file_path, Gm1=self.Gm1, b1=self.b1, Gm2=self.Gm2, b2=self.b2)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        """Loads model weights from a file."""
        data = np.load(file_path)
        self.Gm1 = data['Gm1']
        self.b1 = data['b1']
        self.Gm2 = data['Gm2']
        self.b2 = data['b2']
        print(f"Model loaded from {file_path}")
