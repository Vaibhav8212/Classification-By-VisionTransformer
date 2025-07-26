# 🎥 Vision Transformers (ViT) for Video Classification

---

>  “Video Classification Using CNN with Attention”

While the core project uses CNNs and spatial attention to classify videos into three categories — **General**, **Obscene**, and **Violent** — this extension explores the application of **Vision Transformers (ViT)** to enhance global context understanding.

---

##  Why Vision Transformers?

CNNs are great at learning **local patterns**, but often miss out on the **bigger picture**.

ViTs treat an image as a **sequence of patches** (like words in a sentence), and use **self-attention mechanisms** to capture **long-range relationships**. This makes them highly effective for:
- Complex backgrounds
- Scattered visual cues
- Global classification decisions

---

##  How It Works (Simplified)

1. 🔹 **Patchify the Frame**: Split frame into 16x16 patches  
2. 🔹 **Linear Embedding + Position Encoding**  
3. 🔹 **Pass through Transformer Encoder**  
4. 🔹 **Classification via MLP Head**

Instead of relying on convolution, the model attends to **all parts of the image** equally, which helps in detecting sensitive content that's **not localized**.

---

##  Code Integration (HuggingFace Example)

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load pretrained ViT model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Preprocess input frame
image = Image.open("sample.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = probs.argmax().item()

```

---

ViT in My Project
✔ Replaced CNN with a pretrained ViT model
✔ Used patch-based embeddings for input frames
✔ Fine-tuned on custom 3-class dataset: general, obscene, violent
✔ Aggregated predictions across frames using:

🔸 Average probability

🔸 Threshold voting logic
✔ Integrated within existing Streamlit dashboard

##  Future Directions
 Combine ViT with LSTM / 3D CNN for motion-aware classification

Fuse audio + video for multimodal predictions

 Deploy on cloud with analytics dashboard

 Train on YouTube-scale datasets for real-world generalization

## Dataset Reference
Kaggle: Real Life Violence Situations

HuggingFace: NSFW Detect

 ## References
Dosovitskiy et al., 2020 - An Image is Worth 16x16 Words

HuggingFace Transformers: https://huggingface.co/docs/transformers

TensorFlow Hub ViT: https://tfhub.dev/s?module-type=image-feature-vector&q=vit


