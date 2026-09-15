"""
Minimal single-image predict interface — this is the interface the organizers
will call for grading (Section 4.1: "a minimal interface ... that runs a
prediction on a new image").

Usage:
    python src/predict.py --checkpoint ./checkpoints/best_model.pt --image path/to/image.jpg
"""
import argparse
import json

import torch
from PIL import Image
from torchvision import transforms

from model import get_model


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = get_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # ImageFolder indexes classes alphabetically, so FAKE isn't always index 1.
    # Read the actual mapping saved at training time instead of assuming.
    class_to_idx = ckpt.get("class_to_idx", {"REAL": 0, "FAKE": 1})
    fake_idx = {k.upper(): v for k, v in class_to_idx.items()}.get("FAKE", 1)
    return model, ckpt["img_size"], fake_idx


def predict_image(model, img_size: int, image_path: str, device, fake_idx: int = 1, threshold: float = 0.5):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze(1)
        prob = torch.sigmoid(logit).item()
        prob_fake = prob if fake_idx == 1 else 1 - prob

    label = "AI-generated" if prob_fake >= threshold else "real"
    confidence = prob_fake if prob_fake >= threshold else 1 - prob_fake

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "raw_score": round(prob_fake, 4),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size, fake_idx = load_model(args.checkpoint, device)
    result = predict_image(model, img_size, args.image, device, fake_idx, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
