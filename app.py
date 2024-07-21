import streamlit as st
import subprocess
import os


# Streamlit UI
st.title("PACGAN Urban Landscape Image Generator")
st.write("Generate urban landscape images using Progressive Auxiliary Conditional GAN")

# Input fields
# img_size = st.number_input("Image Size", min_value=4, max_value=256, value=32, step=4)
n_images_clear = st.number_input("Number of Clear Images", min_value=1, max_value=1000, value=10)
n_images_misty = st.number_input("Number of Misty Images", min_value=1, max_value=1000, value=10)
images_path = st.text_input("Path to Save Generated Images", value="path/to/save/images")
device = st.selectbox("Device", options=["cpu", "cuda"])
gpus = st.text_input("GPUs", value="0")

# Hardcoded path to generator model
model_path = "path/to/generator.pth"

if st.button("Generate Images"):
    # Ensure output directory exists
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Generate the images
    cmd = (
        f"python Generator.py "
        f"--img_size {256} "
        f"--n_images_xCLASS {n_images_clear} {n_images_misty} "
        f"--images_path {images_path} "
        f"--device {device} "
        f"--gpus {gpus} "
    )
    subprocess.run(cmd, shell=True)
    
    # Calculate FID score only if image saving is confirmed
    if os.listdir(images_path):  # Check if images are saved
        st.write("Images generated and saved successfully!")
    else:
        st.write("No images found in the folder. Please check the path and try again.")

