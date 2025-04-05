# frontend/draw_canvas.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from src.config import CANVAS_SIZE

def draw_canvas(key="canvas"):
    return st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=15,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key=key,
        update_streamlit=True,
    )

def preprocess_canvas_image(canvas_image_data):
    img = Image.fromarray((255 - canvas_image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28)).convert("L")
    return np.array(img).astype(np.float32) / 255.0
