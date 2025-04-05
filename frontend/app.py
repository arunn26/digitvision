import os
import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Make sure Python can find the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ann_model import ImprovedAnalogNeuralNetwork, pixels_to_voltages
from src.data_loader import get_data_loaders
from src.utils import plot_metrics
from frontend.draw_canvas import draw_canvas, preprocess_canvas_image
from src.config import CANVAS_SIZE, DEFAULT_HIDDEN_SIZE

# ---------------- Page Config ----------------
st.set_page_config(page_title="DigitVision", layout="wide")

# ---------------- Session Init ----------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "train_metrics" not in st.session_state:
    st.session_state.train_metrics = None
if "test_accuracy" not in st.session_state:
    st.session_state.test_accuracy = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_default"

# ---------------- Sidebar ----------------
st.sidebar.header("Model Options")
load_trained = st.sidebar.checkbox("Load Trained Model", value=False)
retrain_model = st.sidebar.checkbox("Retrain Model", value=False)

if retrain_model:
    st.sidebar.subheader("Hyperparameters")
    hidden_size = st.sidebar.slider("Hidden Size", 50, 300, DEFAULT_HIDDEN_SIZE, step=10)
    epochs = st.sidebar.number_input("Epochs", 1, 50, 10)
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
else:
    hidden_size = DEFAULT_HIDDEN_SIZE
    epochs = 10
    learning_rate = 0.001

# ---------------- Load Model & Data ----------------
train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
ann = ImprovedAnalogNeuralNetwork(input_size=784, hidden_size=hidden_size, output_size=10)

# ---------------- Load Trained Model & Metrics ----------------
model_path = "models/model_weights.npz"
metrics_path = "models/training_metrics.npz"

if load_trained:
    if os.path.exists(model_path):
        ann.load(model_path)
        st.sidebar.success("✅ Model weights loaded.")
        st.session_state.model_loaded = True
    else:
        st.sidebar.error("❌ No trained model weights found at 'models/model_weights.npz'.")
        st.session_state.model_loaded = False

    if os.path.exists(metrics_path):
        metrics_data = np.load(metrics_path)
        st.session_state.train_metrics = {
            "train_losses": metrics_data["train_losses"].tolist(),
            "val_losses": metrics_data["val_losses"].tolist(),
            "train_accuracies": metrics_data["train_accuracies"].tolist(),
            "val_accuracies": metrics_data["val_accuracies"].tolist()
        }
        st.session_state.test_accuracy = float(metrics_data["test_accuracy"])
    else:
        st.sidebar.warning("ℹ️ No training metrics found. Retrain to generate metrics.")

# Show status in main area
if load_trained and os.path.exists(model_path):
    st.info("🧠 Pretrained model is loaded and ready.")

# ---------------- Retraining ----------------
if retrain_model and st.sidebar.button("Start Retraining"):
    st.subheader("Training Log")

    # Placeholders for real-time updates
    log_placeholder = st.empty()
    progress_placeholder = st.sidebar.empty()
    status_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Initialize logs
    st.session_state.logs = []

    # Set up a progress bar
    progress_bar = progress_placeholder.progress(0)

    # We'll record epoch-wise losses/accuracies, so we can chart them in real-time
    train_losses_list = []
    val_losses_list = []
    train_accuracies_list = []
    val_accuracies_list = []

    def update_log(msg):
        """Callback to update the log text in real-time."""
        st.session_state.logs.append(msg)
        log_placeholder.text("\n".join(st.session_state.logs))

    # Overriding the train loop to add live chart
    from src.ann_model import pixels_to_voltages

    # Actual training loop
    for epoch_i in range(epochs):
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        # TRAIN PHASE
        for images, labels in train_loader:
            images = images.view(-1, 28*28).numpy()
            labels_onehot = np.eye(ann.output_size)[labels.numpy()]

            for X_pixels, y_onehot in zip(images, labels_onehot):
                X_voltages = pixels_to_voltages(X_pixels)
                output = ann.forward(X_voltages)
                loss = -np.sum(y_onehot * np.log(output + 1e-10))
                total_train_loss += loss

                ann.backward(X_voltages, y_onehot, output, learning_rate=learning_rate)

                if np.argmax(output) == np.argmax(y_onehot):
                    train_correct += 1
                train_total += 1

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total

        # VALIDATION PHASE
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        for images, labels in val_loader:
            images = images.view(-1, 28*28).numpy()
            labels_onehot = np.eye(ann.output_size)[labels.numpy()]

            for X_pixels, y_onehot in zip(images, labels_onehot):
                X_voltages = pixels_to_voltages(X_pixels)
                output = ann.forward(X_voltages)
                loss = -np.sum(y_onehot * np.log(output + 1e-10))
                total_val_loss += loss

                if np.argmax(output) == np.argmax(y_onehot):
                    val_correct += 1
                val_total += 1

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total

        train_losses_list.append(avg_train_loss)
        val_losses_list.append(avg_val_loss)
        train_accuracies_list.append(train_accuracy)
        val_accuracies_list.append(val_accuracy)

        # Update log
        log_msg = (
            f"Epoch {epoch_i+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )
        update_log(log_msg)

        # Update progress bar
        progress = int(100 * (epoch_i + 1) / epochs)
        progress_bar.progress(progress)

        # Live chart
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot Loss
        axs[0].plot(range(1, epoch_i + 2), train_losses_list, label="Train Loss")
        axs[0].plot(range(1, epoch_i + 2), val_losses_list, label="Val Loss")
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Accuracy
        axs[1].plot(range(1, epoch_i + 2), train_accuracies_list, label="Train Acc")
        axs[1].plot(range(1, epoch_i + 2), val_accuracies_list, label="Val Acc")
        axs[1].set_title("Accuracy (%)")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        chart_placeholder.pyplot(fig)
        plt.close(fig)

    # After training completes
    if not os.path.exists("models"):
        os.makedirs("models")
    ann.save(model_path)

    # Evaluate on test set
    test_accuracy = ann.evaluate(test_loader)
    st.session_state.test_accuracy = test_accuracy
    st.session_state.train_metrics = {
        "train_losses": train_losses_list,
        "val_losses": val_losses_list,
        "train_accuracies": train_accuracies_list,
        "val_accuracies": val_accuracies_list
    }
    np.savez(
        metrics_path,
        train_losses=np.array(train_losses_list),
        val_losses=np.array(val_losses_list),
        train_accuracies=np.array(train_accuracies_list),
        val_accuracies=np.array(val_accuracies_list),
        test_accuracy=test_accuracy
    )

    status_placeholder.success(f"🎉 Model retrained successfully! Final Test Accuracy: {test_accuracy:.2f}%")

# ---------------- Layout ----------------
spacer1, draw_col, pred_col, spacer2 = st.columns([0.5, 0.9, 0.8, 0.5])

with draw_col:
    st.subheader("Draw a Digit")
    canvas_result = draw_canvas(key=st.session_state.canvas_key)
    if st.button("Clear Drawing"):
        st.session_state.canvas_key = str(np.random.rand())
        st.rerun()

with pred_col:
    st.subheader("Prediction")

    if canvas_result.image_data is not None:
        img_np = preprocess_canvas_image(canvas_result.image_data)

        if np.sum(img_np) < 10:
            st.warning("Please draw a digit before predicting.")
        else:
            st.image(img_np, width=100, caption="28x28 Input", clamp=True)
            X_voltages = pixels_to_voltages(img_np.flatten())

            if st.button("Predict"):
                if load_trained or retrain_model:
                    output_probs = ann.forward(X_voltages)
                    predicted_digit = int(np.argmax(output_probs))
                    st.markdown(f"**Predicted Digit:** `{predicted_digit}`")

                    fig, ax = plt.subplots()
                    ax.bar(range(10), output_probs)
                    ax.set_xticks(range(10))
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Confidence")
                    st.pyplot(fig)
                else:
                    st.warning("Please load or retrain the model first.")

    st.button("Clear Prediction", key="clear_prediction", on_click=st.rerun)

# ---------------- Metrics ----------------
if st.session_state.train_metrics:
    st.subheader("Training Metrics")
    fig = plot_metrics(
        st.session_state.train_metrics["train_losses"],
        st.session_state.train_metrics["val_losses"],
        st.session_state.train_metrics["train_accuracies"],
        st.session_state.train_metrics["val_accuracies"]
    )
    st.pyplot(fig)

    if st.session_state.test_accuracy is not None:
        st.markdown(f"### Test Accuracy: `{st.session_state.test_accuracy:.2f}%`")
