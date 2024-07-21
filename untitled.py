import os
import shutil
import csv
from PIL import Image

# Directories containing the original images
misty_dir = r'C:\Users\Bhuhpramaan\Supriya\Data\Misty'
clear_dir = r'C:\Users\Bhuhpramaan\Supriya\Data\CLear'

# Directory to store mixed images
mixed_dir = 'data/mixed_images'
os.makedirs(mixed_dir, exist_ok=True)

# CSV file to store image IDs and classes
csv_file = 'data/mixed_labels.csv'

# Desired image size for preprocessing
image_size = (256, 256)

# Function to preprocess and copy images to the mixed directory and create the CSV file
def prepare_mixed_data(misty_dir, clear_dir, mixed_dir, csv_file, image_size):
    image_id = 0
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Class'])
        
        for img_name in os.listdir(clear_dir):
            src_path = os.path.join(clear_dir, img_name)
            dst_path = os.path.join(mixed_dir, f"{image_id}.png")
            preprocess_and_copy_image(src_path, dst_path, image_size)
            writer.writerow([image_id, 0])
            image_id += 1
        
        for img_name in os.listdir(misty_dir):
            src_path = os.path.join(misty_dir, img_name)
            dst_path = os.path.join(mixed_dir, f"{image_id}.png")
            preprocess_and_copy_image(src_path, dst_path, image_size)
            writer.writerow([image_id, 1])
            image_id += 1

# Function to preprocess (resize) and copy an image
def preprocess_and_copy_image(src_path, dst_path, image_size):
    with Image.open(src_path) as img:
        img = img.resize(image_size, Image.ANTIALIAS)
        img.save(dst_path)

# Run the function
prepare_mixed_data(misty_dir, clear_dir, mixed_dir, csv_file, image_size)

print("Mixed data preparation complete.")
