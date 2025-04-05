import os
# Disable Streamlit's file watcher to avoid torch.classes errors on Python 3.12
os.environ["STREAMLIT_WATCHED_FILES"] = "[]"

import sys, base64
from io import BytesIO
# Ensure the project root is added to the Python path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Import st_canvas from streamlit-drawable-canvas (not used now)
#from streamlit_drawable_canvas import st_canvas

from src.ann_model import ImprovedAnalogNeuralNetwork, pixels_to_voltages
from src.data_loader import get_data_loaders
from src.utils import plot_metrics

# Define file paths for model weights and training metrics.
MODEL_PATH = "model_weights.npz"
METRICS_PATH = "training_metrics.npz"

# Initialize session state for logs, metrics, test accuracy if not already set.
if "logs" not in st.session_state:
    st.session_state.logs = []
if "train_metrics" not in st.session_state:
    st.session_state.train_metrics = None
if "test_accuracy" not in st.session_state:
    st.session_state.test_accuracy = None

# ----- Custom CSS for Responsive Layout -----
st.markdown(
    """
    <style>
    /* General styles */
    .main {
        background-color: #f0f2f6;
        padding: 1rem;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
    }
    /* Sidebar customization */
    .css-1d391kg {  /* Streamlit's sidebar container */
        width: 300px;
    }
    /* Responsive adjustments for small screens */
    @media screen and (max-width: 768px) {
        .css-1d391kg {
            width: 100% !important;
        }
        .stApp {
            margin: 0;
        }
    }
    /* Headings color */
    h1, h2, h3 {
        color: #2e7bcf;
    }
    /* Button customization */
    .stButton>button {
        background-color: #2e7bcf;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- App Title -----
st.title("DigitVision: Analog Neural Network Explorer")

# ----- Sidebar: Model Options -----
st.sidebar.header("Model Options")
load_trained = st.sidebar.checkbox("Load Trained Model", value=False)
retrain_model = st.sidebar.checkbox("Retrain Model", value=False)

if retrain_model:
    st.sidebar.subheader("Hyperparameter Tuning")
    hidden_size = st.sidebar.slider("Hidden Layer Size", min_value=50, max_value=300, value=100, step=10)
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=50, value=10)
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
else:
    hidden_size = 100
    epochs = 10
    learning_rate = 0.001

# ----- Data Loaders and Model Initialization -----
train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
ann = ImprovedAnalogNeuralNetwork(input_size=784, hidden_size=hidden_size, output_size=10)

# ----- Load Trained Model and Metrics if Available -----
if load_trained:
    if os.path.exists(MODEL_PATH):
        ann.load(MODEL_PATH)
        st.sidebar.success("Trained model loaded.")
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
        else:
            st.sidebar.info("No training metrics file found.")
    else:
        st.sidebar.error("No trained model found. Please retrain.")

# ----- Retraining Section with Persistent Logs, Graph, and Test Accuracy -----
if retrain_model:
    if st.sidebar.button("Retrain Model"):
        st.write("Retraining model... Please be patient, this may take several minutes.")
        log_placeholder = st.empty()
        st.session_state.logs = []
        def update_log(message):
            st.session_state.logs.append(message)
            log_placeholder.text("\n".join(st.session_state.logs))
        train_losses, val_losses, train_accuracies, val_accuracies = ann.train(
            train_loader, val_loader, epochs=epochs, learning_rate=learning_rate, log_callback=update_log
        )
        st.success("Model retrained successfully!")
        ann.save(MODEL_PATH)
        test_accuracy = ann.evaluate(test_loader)
        st.session_state.test_accuracy = test_accuracy
        st.write(f"**Test Accuracy:** {test_accuracy:.2f}%")
        st.session_state.train_metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }
        np.savez(METRICS_PATH,
                 train_losses=np.array(train_losses),
                 val_losses=np.array(val_losses),
                 train_accuracies=np.array(train_accuracies),
                 val_accuracies=np.array(val_accuracies),
                 test_accuracy=test_accuracy)

# ----- Main Interface Tabs -----
tabs = st.tabs(["Home", "Trained Metrics", "Sample Prediction", "Draw Digit (Coming Soon)"])

with tabs[0]:
    st.header("Welcome to DigitVision")
    st.write("""
        **Overview:**  
        DigitVision is an interactive explorer for a custom analog neural network designed for handwritten digit recognition using the MNIST dataset.  
        
        **Key Features:**  
        - **Load or Retrain the Model:** Use the sidebar to load a pre-trained model or retrain the network with custom hyperparameters.
        - **View Training Metrics:** Monitor training and validation loss/accuracy over epochs and examine the model's internal weight matrices.
        - **Sample Prediction:** Test the model on randomly selected MNIST test samples.
        - **Draw Digit (Coming Soon):** A future feature will let you draw a digit and see the model's prediction in real time.
        
        **How It Works:**  
        The network converts pixel values into voltage levels and propagates the data through hidden and output layers. Training adjusts the weight matrices (GM1 and GM2) to minimize prediction error.
        
        Enjoy exploring the inner workings of our analog neural network and discovering how it interprets handwritten digits!
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", caption="Sample MNIST Digits", use_container_width=True)

with tabs[1]:
    st.header("Trained Metrics")
    if st.session_state.train_metrics is not None:
        tm = st.session_state.train_metrics
        fig_metrics = plot_metrics(tm["train_losses"], tm["val_losses"], tm["train_accuracies"], tm["val_accuracies"])
        st.pyplot(fig_metrics)
        if st.session_state.test_accuracy is not None:
            st.write(f"**Test Accuracy:** {st.session_state.test_accuracy:.2f}%")
        
        # Button to load and display the weight matrices.
        if st.button("Load Weights"):
            st.subheader("Model Weights")
            with st.expander("Show GM1 Weights"):
                st.dataframe(ann.Gm1)
            with st.expander("Show GM2 Weights"):
                st.dataframe(ann.Gm2)
    else:
        st.info("No training metrics available. Please retrain the model using the sidebar options.")

with tabs[2]:
    st.header("Sample Prediction from Test Data")
    if st.button("Show a Sample Prediction", key="sample_pred"):
        images, labels = next(iter(test_loader))
        idx = random.randint(0, len(labels) - 1)
        image = images[idx].view(-1, 28 * 28).numpy().flatten()
        true_label = labels[idx].item()
        X_voltages = pixels_to_voltages(image)
        predicted_label = ann.predict(X_voltages)
        st.write(f"**True Label:** {true_label}")
        st.write(f"**Predicted Label:** {predicted_label}")
        image_28 = image.reshape(28, 28)
        image_pil = Image.fromarray((image_28 * 255).astype(np.uint8))
        image_pil_half = image_pil.resize((14, 14), resample=Image.LANCZOS)
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(image_pil_half, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

with tabs[3]:
    st.header("Draw Digit (Coming Soon)")
    st.info("This feature is under development and will be available in a future update.")
