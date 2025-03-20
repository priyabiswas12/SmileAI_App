import streamlit as st
import os
import numpy as np
import torch
import torch.nn as nn
from predict.utils.meshsegnet import *
from vedo import *
import pandas as pd
from predict.utils.loss_and_metrics import *
from scipy.spatial import distance_matrix
import scipy.io as sio
import shutil
import time
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph
from predict.inference import Inference
from PIL import Image


def get_image_path_for_label(label):
    image_directory = "./ui_images/lower_stock"  # Folder containing images
    #print(os.path.join(image_directory, f"{label}.jpg"))
    return os.path.join(image_directory, f"{label}.jpg")




st.title("3D STL Model Processor")

uploaded_file_lower = st.file_uploader("Upload a Lower Jaw STL file", type=["stl"],key="file1")



if uploaded_file_lower:
    # Save file to uploads directory
    file_path = os.path.join("./predict/uploads/lower", uploaded_file_lower.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_lower.getbuffer())
    st.success(f"File uploaded: {uploaded_file_lower.name}")

    p=Inference()
    labels= p.predict_labels_lower(uploaded_file_lower.name)
    print(labels)



    #st.write(f"Detected Labels: {labels}")
    num_teeth = 16
    tooth_positions = list(range(1, num_teeth + 1))

    st.title("Mandibular Teeth Display")

    # Create 16 columns for 16 teeth
    cols = st.columns(16)
    IMAGE_WIDTH = 100  # Adjust width as needed
    IMAGE_HEIGHT = 150

    
    
    for index, tooth_id in enumerate(tooth_positions):
        image_path = get_image_path_for_label(tooth_id)

        img = Image.open(image_path)
        img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize to fixed bounding box

        with cols[index]:

            if tooth_id in labels and os.path.exists(image_path):  # Each tooth gets its own column
                st.image(img_resized, caption=f"{index+1}", use_container_width=True)
            else:
                cols[index].empty()





    # cols = st.columns(num_teeth)
    # for i, tooth_id in enumerate(tooth_positions):
    #     image_path = get_image_path_for_label(tooth_id)
        
    #     if tooth_id in labels and os.path.exists(image_path):
    #         cols[i].image(image_path, caption=f"Tooth {tooth_id}", use_container_width=True)
    #     else:
    #         cols[i].empty()





    # for label in labels:
    #     image_path = get_image_path_for_label(label)
    #     if os.path.exists(image_path):
    #         st.image(image_path, caption=f"Image for Label {label}", use_column_width=True)
    #     else:
    #         st.warning(f"Image for Label {label} not found!")




uploaded_file_upper = st.file_uploader("Upload a Upper Jaw STL file", type=["stl"], key="file2")
if uploaded_file_upper:
    # Save file to uploads directory
    file_path = os.path.join("./predict/uploads/upper", uploaded_file_upper.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_upper.getbuffer())
    st.success(f"File uploaded: {uploaded_file_upper.name}")

    p=Inference()
    labels= p.predict_labels_upper(uploaded_file_upper.name)
    print(labels)