# ============================================================
#  Pretrained CNN Image Classification — Lab 10.9.4 Style
#  Using ResNet50 pretrained on ImageNet
#  Python implementation for Google Colab
# ============================================================

# ---- Install dependencies (run once) ----
# !pip install tensorflow Pillow requests

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from io import BytesIO

import tensorflow as tf
import tensorflow.keras.applications
from tensorflow.keras.applications.resnet50 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image as keras_image

print(f"TensorFlow version: {tf.__version__}")

# ------------------------------------------------------------
# 1. Load pretrained ResNet50
#    weights='imagenet': pretrained on 1.2M ImageNet images
#    include_top=True:   keep the 1000-class classifier head
# ------------------------------------------------------------
model = tensorflow.keras.applications.ResNet50(weights='imagenet', include_top=True)
print("ResNet50 loaded — input shape:", model.input_shape)

# ------------------------------------------------------------
# 2. Define 10 animal images (Wikipedia public domain)
# ------------------------------------------------------------
animals = [
    {
        "name": "Dog (Labrador)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg"
    },
    {
        "name": "Cat",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/640px-Cat_November_2010-1a.jpg"
    },
    {
        "name": "Elephant",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/640px-African_Bush_Elephant.jpg"
    },
    {
        "name": "Eagle",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/640px-24701-nature-natural-beauty.jpg"
    },
    {
        "name": "Horse",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Nokota_Horses_cropped.jpg/640px-Nokota_Horses_cropped.jpg"
    },
    {
        "name": "Lion",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Lion_waiting_in_Namibia.jpg/640px-Lion_waiting_in_Namibia.jpg"
    },
    {
        "name": "Penguin",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/South_Shetland-2016-Deception_Island%E2%80%93Chinstrap_penguin_%28Pygoscelis_antarctica%29_04.jpg/640px-South_Shetland-2016-Deception_Island%E2%80%93Chinstrap_penguin_%28Pygoscelis_antarctica%29_04.jpg"
    },
    {
        "name": "Cow",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Cow_female_black_white.jpg/640px-Cow_female_black_white.jpg"
    },
    {
        "name": "Giant Panda",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/640px-Grosser_Panda.JPG"
    },
    {
        "name": "Parrot (Macaw)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Scarlet-Macaw.jpg/640px-Scarlet-Macaw.jpg"
    }
]

# ------------------------------------------------------------
# 3. Helper: load image from URL and preprocess for ResNet50
# ------------------------------------------------------------
def load_and_preprocess(url):
    """
    Downloads image from URL, resizes to 224x224,
    and applies ResNet50 preprocessing.
    Returns: preprocessed array (1, 224, 224, 3)
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = keras_image.img_to_array(img)           # (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)             # (1, 224, 224, 3)
    arr = preprocess_input(arr)                   # ResNet50 normalization
    return img, arr

# ------------------------------------------------------------
# 4. Classify all 10 images
# ------------------------------------------------------------
print("\n" + "="*60)
print("   CNN Predictions: Top 5 Classes per Animal Image")
print("   Model: ResNet50 pretrained on ImageNet")
print("="*60 + "\n")

results     = []
pil_images  = []

for i, animal in enumerate(animals):
    print(f"Image {i+1}: {animal['name'].upper()}")
    print("-" * 50)

    try:
        pil_img, arr = load_and_preprocess(animal["url"])
        preds        = model.predict(arr, verbose=0)
        top5         = decode_predictions(preds, top=5)[0]

        pil_images.append(pil_img)
        results.append({
            "name": animal["name"],
            "top5": top5
        })

        for rank, (class_id, class_name, prob) in enumerate(top5, 1):
            print(f"  {rank}. {class_name:<30s} {prob*100:.2f}%")

    except Exception as e:
        print(f"  ERROR: {e}")
        pil_images.append(None)
        results.append({"name": animal["name"], "top5": None})

    print()

# ------------------------------------------------------------
# 5. Summary table
# ------------------------------------------------------------
print("="*60)
print("   Summary: Top Predicted Class per Image")
print("="*60)
print(f"{'Image':<20} {'Top Class':<30} {'Probability':>12}")
print("-"*60)
for r in results:
    if r["top5"] is not None:
        top_class = r["top5"][0][1]
        top_prob  = r["top5"][0][2]
        print(f"{r['name']:<20} {top_class:<30} {top_prob*100:>11.2f}%")
print("="*60)

# ------------------------------------------------------------
# 6. Plot: 2x5 grid — image + top 5 predictions as bar chart
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(20, 9))
axes = axes.flatten()

colors = ["#2196F3", "#42A5F5", "#90CAF9", "#BBDEFB", "#E3F2FD"]

for i, (result, pil_img) in enumerate(zip(results, pil_images)):
    ax = axes[i]

    if pil_img is not None and result["top5"] is not None:
        # Show image as background (small inset)
        ax.imshow(pil_img, aspect="auto", alpha=0.15,
                  extent=[0, 1, 0, 1], transform=ax.transAxes)

        # Bar chart of top 5 probabilities
        top5      = result["top5"]
        classes   = [t[1].replace("_", " ") for t in top5]
        probs     = [t[2] * 100 for t in top5]

        bars = ax.barh(range(5), probs,
                       color=colors, edgecolor="white",
                       height=0.65)

        # Labels
        ax.set_yticks(range(5))
        ax.set_yticklabels(classes, fontsize=8.5)
        ax.invert_yaxis()   # top prediction at top
        ax.set_xlabel("Probability (%)", fontsize=8)
        ax.set_xlim(0, max(probs) * 1.25)

        # Annotate bars with percentage
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{prob:.1f}%",
                    va="center", ha="left", fontsize=7.5, fontweight="bold")

        ax.set_title(result["name"], fontsize=10,
                     fontweight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)

    else:
        ax.text(0.5, 0.5, "Failed to load",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(result["name"], fontsize=10)

plt.suptitle("ResNet50 (ImageNet): Top 5 Predictions per Animal Image",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("cnn_animal_predictions.png",
            dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("\nPlot saved: cnn_animal_predictions.png")