from PIL import Image
import matplotlib.pyplot as plt

image_folder = "images/"
image_filename = "192473.jpg"
image_path = image_folder + image_filename

# Load the image
img = Image.open(image_path)

# Show the image
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()
