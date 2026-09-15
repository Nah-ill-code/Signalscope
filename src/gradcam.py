"""
Grad-CAM explanation for a single verdict — Bonus Module A (the headline bonus).

Usage:
    python src/gradcam.py --checkpoint ./checkpoints/best_model.pt \
        --image path/to/image.jpg --img-size 224 --out heatmap.png

Remember (Section 4.3 scoring rubric): describe only what the heat-map actually
shows (which region lit up), don't invent artefact claims the map doesn't support,
and communicate uncertainty honestly ("likely", not "certain").
"""
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import get_model, get_last_conv_layer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, fake_idx: int = 1):
        self.model.zero_grad()
        logit = self.model(input_tensor).squeeze(1)
        logit.backward()

        pooled_grads = torch.mean(self.gradients, dim=(0, 2, 3))
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        prob = torch.sigmoid(logit).item()
        prob_fake = prob if fake_idx == 1 else 1 - prob
        return heatmap, prob_fake


def overlay_heatmap(image_pil: Image.Image, heatmap: np.ndarray, img_size: int) -> np.ndarray:
    image = np.array(image_pil.resize((img_size, img_size)))
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * image + 0.45 * heatmap_colored)
    return overlay


def analyze_heatmap(heatmap: np.ndarray, threshold: float = 0.6) -> dict:
    """Compute a grounded description of WHERE the model's attention actually
    concentrated, based on the real Grad-CAM activations. This deliberately
    does not guess at a specific artifact type (texture, lighting, geometry)
    since a heat-map alone can't verify that - it only shows location and
    concentration, which is what we report."""
    h, w = heatmap.shape
    hot_mask = (heatmap >= threshold).astype(np.uint8)
    hot_fraction = float(hot_mask.sum()) / hot_mask.size

    contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"concentrated": False, "region": None, "hot_fraction": hot_fraction}

    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return {"concentrated": False, "region": None, "hot_fraction": hot_fraction}

    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    row = "upper" if cy < h / 3 else ("lower" if cy > 2 * h / 3 else "middle")

    if row == "middle" and col == "center":
        region = "the center of the image"
    elif row == "middle":
        region = f"the {col} side of the image"
    elif col == "center":
        region = f"the {row} part of the image"
    else:
        region = f"the {row}-{col} area of the image"

    return {
        "concentrated": hot_fraction < 0.35,
        "region": region,
        "hot_fraction": hot_fraction,
    }


def explanation_bullets(analysis: dict, is_fake: bool) -> list:
    """Turn the heat-map analysis into a short, honest explanation.
    Grounded in what was actually measured - no invented defect categories."""
    bullets = []
    region = analysis.get("region")
    concentrated = analysis.get("concentrated")
    pct = round(analysis.get("hot_fraction", 0) * 100)

    if region and concentrated:
        bullets.append(f"Model attention concentrated in {region} (~{pct}% of the frame highlighted)")
        if is_fake:
            bullets.append("This localized pattern is where the model found the strongest evidence for its verdict")
        else:
            bullets.append("No comparable anomaly was found elsewhere in the image")
    elif region:
        bullets.append(f"Model attention was spread broadly, with a mild peak toward {region}")
        bullets.append("Broad, low-concentration attention is typical when no single region stands out")
    else:
        bullets.append("Model attention was spread broadly with no single concentrated region")

    return bullets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out", default="heatmap.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = get_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_to_idx = ckpt.get("class_to_idx", {"REAL": 0, "FAKE": 1})
    fake_idx = {k.upper(): v for k, v in class_to_idx.items()}.get("FAKE", 1)

    target_layer = get_last_conv_layer(model, ckpt["backbone"])
    cam = GradCAM(model, target_layer)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    img = Image.open(args.image).convert("RGB")
    input_tensor = tf(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    heatmap, prob_fake = cam.generate(input_tensor, fake_idx)
    label = "AI-generated" if prob_fake >= 0.5 else "REAL"
    confidence = prob_fake if prob_fake >= 0.5 else 1 - prob_fake

    overlay = overlay_heatmap(img, heatmap, args.img_size)
    Image.fromarray(overlay).save(args.out)

    print(f"Verdict: likely {label} (confidence {confidence:.2f})")
    print(f"Heat-map saved to {args.out}")
    print("Reminder: describe only the highlighted region in your explanation text — "
          "don't claim artefacts the map doesn't actually show.")


if __name__ == "__main__":
    main()
