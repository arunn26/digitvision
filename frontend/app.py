import os
os.environ["STREAMLIT_WATCHED_FILES"] = "[]"

import sys
import random
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ann_model import ImprovedAnalogNeuralNetwork, pixels_to_voltages
from src.data_loader import get_data_loaders
from src.utils import plot_metrics
from frontend.draw_canvas import draw_canvas, preprocess_canvas_image

MODEL_PATH = "models/model_weights.npz"
METRICS_PATH = "models/training_metrics.npz"

st.set_page_config(page_title="DigitVision", layout="wide")

if "train_metrics" not in st.session_state:
    st.session_state.train_metrics = None
if "test_accuracy" not in st.session_state:
    st.session_state.test_accuracy = None
if "epochs_trained" not in st.session_state:
    st.session_state.epochs_trained = None

train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
ann = ImprovedAnalogNeuralNetwork(input_size=784, hidden_size=100, output_size=10)

if os.path.exists(MODEL_PATH):
    ann.load(MODEL_PATH)

    if os.path.exists(METRICS_PATH):
        metrics_data = np.load(METRICS_PATH)
        st.session_state.train_metrics = {
            "train_losses": metrics_data["train_losses"].tolist(),
            "val_losses": metrics_data["val_losses"].tolist(),
            "train_accuracies": metrics_data["train_accuracies"].tolist(),
            "val_accuracies": metrics_data["val_accuracies"].tolist()
        }
        if "test_accuracy" in metrics_data:
            st.session_state.test_accuracy = float(metrics_data["test_accuracy"])
        st.session_state.epochs_trained = len(metrics_data["train_losses"])
    else:
        st.warning("Training metrics file not found.")
else:
    st.warning("Trained model not found. Please train it first.")

st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
    padding: 1rem;
}
.stApp {
    max-width: 100%;
    margin: 0 auto;
}
h1, h2, h3 {
    color: #2e7bcf;
}
.stButton>button {
    background-color: #2e7bcf;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Navigation")
page_choice = st.sidebar.radio(
    "Go to:",
    ["Home", "Sample Prediction", "Draw Digit", "Metrics & Weights"]
)

st.title("DigitVision: Single-Page Explorer")

if page_choice == "Home":
    st.header("Welcome to DigitVision")
    st.write("""
    **Overview**  
    1. Home – Intro and MNIST examples  
    2. Sample Prediction – Try the model on random digits  
    3. Draw Digit – Draw your own digit and predict  
    4. Metrics & Weights – View training performance
    """)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
        caption="MNIST Examples",
        use_container_width=True,
    )

elif page_choice == "Sample Prediction":
    st.header("Sample Prediction from MNIST Data Set")
    if st.button("Get a Random Sample"):
        images, labels = next(iter(test_loader))
        idx = random.randint(0, len(labels) - 1)
        image = images[idx].view(-1, 28 * 28).numpy().flatten()
        true_label = labels[idx].item()

        X_voltages = pixels_to_voltages(image)
        predicted_label = ann.predict(X_voltages)

        st.write(f"**True Label:** {true_label}")
        st.write(f"**Predicted Label:** {predicted_label}")

        image_28 = image.reshape(28, 28)
        fig, ax = plt.subplots()
        ax.imshow(image_28, cmap="gray")
        ax.set_title("Random MNIST Sample")
        ax.axis("off")
        st.pyplot(fig)

elif page_choice == "Draw Digit":
    st.header("Draw a Digit")

    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas_digit"

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        canvas_result = draw_canvas(
            key=st.session_state.canvas_key,
            stroke_width=15,
        )

    with col_right:
        st.markdown("### Preview")
        if canvas_result.image_data is not None:
            img_np = preprocess_canvas_image(canvas_result.image_data)
            st.image(img_np, width=100, clamp=True)
        else:
            st.write("Draw something to see preview.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear"):
            st.session_state.canvas_key = (
                "canvas_digit_2" if st.session_state.canvas_key == "canvas_digit" else "canvas_digit"
            )
            st.rerun()

    with col2:
        if st.button("Predict"):
            if canvas_result.image_data is not None:
                img_np = preprocess_canvas_image(canvas_result.image_data)
                if np.sum(img_np) < 10:
                    st.warning("Please draw a digit before predicting.")
                else:
                    st.image(img_np, width=100, caption="28x28 Input", clamp=True)
                    X_voltages = pixels_to_voltages(img_np.flatten())
                    output_probs = ann.forward(X_voltages)
                    predicted_digit = int(np.argmax(output_probs))
                    st.markdown(f"**Predicted Digit:** {predicted_digit}")

                    fig, ax = plt.subplots()
                    ax.bar(range(10), output_probs)
                    ax.set_xticks(range(10))
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Confidence")
                    st.pyplot(fig)
            else:
                st.warning("Canvas is empty. Please draw a digit first.")

elif page_choice == "Metrics & Weights":
    

    if st.session_state.train_metrics:
        tm = st.session_state.train_metrics
        epochs = st.session_state.epochs_trained

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Epoch Stats")
            st.write(f"- Train Loss: `{tm['train_losses'][-1]:.4f}`")
            st.write(f"- Validation Loss: `{tm['val_losses'][-1]:.4f}`")
            st.write(f"- Train Accuracy: `{tm['train_accuracies'][-1]:.2f}%`")
            st.write(f"- Validation Accuracy: `{tm['val_accuracies'][-1]:.2f}%`")

        with col2:
            best_val_acc = max(tm["val_accuracies"])
            best_epoch = tm["val_accuracies"].index(best_val_acc) + 1
            st.subheader("Best Model Performance")
            st.write(f"Best Validation Accuracy: `{best_val_acc:.2f}%`")
            st.write(f"Achieved at Epoch: `{best_epoch}`")

        st.markdown("### Accuracy & Loss Curves")
        fig = plot_metrics(
            tm["train_losses"],
            tm["val_losses"],
            tm["train_accuracies"],
            tm["val_accuracies"]
        )
        st.pyplot(fig)
    else:
        st.info("No training metrics available. Train the model first.")

    st.subheader("Gm1 & Gm2 Weights")
    if os.path.exists(MODEL_PATH):
        
        with st.expander("Gm1"):
            st.write(ann.Gm1[:2, :5])
        with st.expander("Gm2"):
            st.write(ann.Gm2[:2, :5])
    else:
        st.warning("Model not found. Please train and save first.")
