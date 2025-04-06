import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from src.config import CANVAS_SIZE


def draw_canvas(key="canvas", stroke_width=15):
    """
    Create a drawable canvas with a light grid background.
    Returns a canvas_result object with .image_data
    """
    return st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # white background
        stroke_width=stroke_width,
        stroke_color="#000000",               # black ink
        background_color="#FFFFFF",           # white background
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key=key,
        update_streamlit=True,
        display_toolbar=True,
    )


def preprocess_canvas_image(canvas_image_data):
    """
    Converts canvas RGBA to 28x28 grayscale, normalizes to 0–1.
    Inverts color (black ink on white bg becomes white-on-black).
    """
    img = Image.fromarray((255 - canvas_image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28)).convert("L")
    return np.array(img).astype(np.float32) / 255.0
