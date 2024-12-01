from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Display Colab badge (if using in Colab)
from IPython.display import display, HTML
display(HTML("""
<a target="_blank" href="https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
"""))

# Check if running in Colab
using_colab = True
if using_colab:
    import torchvision
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

# Display an example image
image_path = "images/dog.jpg"

# Using PIL to open the image
image = Image.open(image_path)
image = np.array(image)  # Convert PIL Image to numpy array

# Convert from BGR to RGB (PIL reads in RGB by default)
image = image[:, :, :3]  # In case the image has an alpha channel, remove it (RGBA -> RGB)

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()

# Load SAM model
sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM model and mask generator
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate masks
masks = mask_generator.generate(image)

# Print some info about the generated masks
print(f"Number of masks generated: {len(masks)}")
print(f"Keys in first mask: {masks[0].keys()}")

# Visualize the masks overlaid on the image
def show_anns(anns):
    if not anns:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # Random color with transparency
        img[m] = color_mask
    ax.imshow(img)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

# Mask generation with custom parameters
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100  # Requires OpenCV for post-processing
)

masks2 = mask_generator_2.generate(image)

print(f"Number of masks generated with custom parameters: {len(masks2)}")

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show()
