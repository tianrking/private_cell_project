import os
from PIL import Image

# Define the target color
target_color = (255, 255, 255)

# Define the replacement color
replacement_color = (128, 128, 128)

# Iterate over all the files in the directory
for filename in os.listdir(r"D:\ww\lwd\data_only\Data\wtr-data_heren\heren-yuanhe\label_all"):
    # Check if the file is an image
    if filename.endswith(".png"):
        # Open the image
        img = Image.open(r"D:\ww\lwd\data_only\Data\wtr-data_heren\heren-yuanhe\label_all/" + filename)

        # Get the image pixels
        pixels = img.load()

        # Replace the target pixels with the replacement color
        for i in range(img.width):
            for j in range(img.height):
                # Get the pixel value
                pixel = pixels[i, j]

                # Check if the pixel matches the target color
                if pixel == target_color:
                    pixels[i, j] = replacement_color

        # Save the modified image
        img.save(r"D:\ww\lwd\data_only\Data\wtr-data_heren\heren-yuanhe\label_all/" + filename)
