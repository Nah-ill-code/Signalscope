"""
Minimal web front-end for SignalScope. Same visual layout as the default
Interface version (image in on the left, Verdict / heat-map / Explanation on
the right, Submit / Clear / Flag buttons), but the buttons are wired manually
so Clear only resets the image - it no longer wipes out the last result.

Usage:
    python app.py --checkpoint ./checkpoints/best_model.pt
"""
import argparse
import csv
import os
import sys
import time

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "src")
from src.gradcam import GradCAM, analyze_heatmap, explanation_bullets, overlay_heatmap
from src.model import get_last_conv_layer, get_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="./checkpoints_v4/best_model.pt",
                    help="Path to a trained checkpoint. Defaults to checkpoints_v4/best_model.pt")
parser.add_argument("--share", action="store_true")
args, _ = parser.parse_known_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.checkpoint, map_location=device)
model = get_model(ckpt["backbone"], pretrained=False).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
img_size = ckpt["img_size"]

class_to_idx = ckpt.get("class_to_idx", {"REAL": 0, "FAKE": 1})
fake_idx = {k.upper(): v for k, v in class_to_idx.items()}.get("FAKE", 1)

target_layer = get_last_conv_layer(model, ckpt["backbone"])
cam = GradCAM(model, target_layer)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize,
])

VERDICT_PLACEHOLDER = "<div class='explanation-title'>Verdict</div>"
EXPLANATION_PLACEHOLDER = "<div class='explanation-title'>Explanation</div>"
FLAG_DIR = "flagged"

CSS = """
h1 {font-size: 32px !important; font-weight: 700 !important;}
.subtitle {color: #a0a0a0 !important; font-size: 14px !important;}

.result-box {
    background: #1b1b1b;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 20px 24px;
}
.result-box .verdict-label {font-size: 22px; font-weight: 700; color: #f5f5f5;}
.result-box .verdict-confidence {font-size: 14px; color: #999999; margin-top: 6px;}

.explanation-box {
    background: #1b1b1b;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 20px 24px;
}
.explanation-box .explanation-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #999999;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.explanation-box ul {list-style: none; margin: 0; padding: 0;}
.explanation-box li {
    font-size: 15px;
    color: #f0f0f0;
    margin-bottom: 10px;
    padding-left: 20px;
    position: relative;
    line-height: 1.4;
}
.explanation-box li::before {
    content: "";
    position: absolute;
    left: 0;
    top: 6px;
    width: 7px;
    height: 7px;
    border: 1.5px solid #999999;
    border-radius: 50%;
}
.explanation-box .disclaimer {font-size: 13px; color: #808080; margin-top: 4px;}

/* Keep "Runs" (history) and "Settings" in the footer, drop the rest */
a[href*="gradio.app"] {display: none !important;}
a[href*="view=api"], a[href*="/docs"], [title="Use via API"], [aria-label="Use via API"] {
    display: none !important;
}
"""


def run(image: Image.Image):
    if image is None:
        return VERDICT_PLACEHOLDER, None, EXPLANATION_PLACEHOLDER

    image = image.convert("RGB")
    input_tensor = tf(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    heatmap, prob_fake = cam.generate(input_tensor, fake_idx)
    is_fake = prob_fake >= 0.5
    confidence = prob_fake if is_fake else 1 - prob_fake
    confidence_pct = f"{confidence * 100:.1f}%"
    label = "Likely AI-generated" if is_fake else "Likely real"

    overlay = overlay_heatmap(image, heatmap, img_size)

    analysis = analyze_heatmap(heatmap)
    bullets = explanation_bullets(analysis, is_fake)
    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)

    verdict_html = (
        f"<div class='verdict-label'>{label}</div>"
        f"<div class='verdict-confidence'>Confidence: {confidence_pct}</div>"
    )
    explanation_html = (
        "<div class='explanation-title'>Explanation</div>"
        f"<ul>{bullets_html}</ul>"
        "<div class='disclaimer'>"
        "This is a likelihood assessment based on model activations, not a certainty."
        "</div>"
    )

    return verdict_html, overlay, explanation_html


def clear_all():
    # Reset everything back to its empty/placeholder state.
    return None, VERDICT_PLACEHOLDER, None, EXPLANATION_PLACEHOLDER


def flag_current(image, verdict_html_value, explanation_html_value):
    if image is None:
        gr.Warning("Nothing to flag yet - run a prediction first.")
        return
    os.makedirs(FLAG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    img_path = os.path.join(FLAG_DIR, f"{ts}.png")
    image.save(img_path)
    log_path = os.path.join(FLAG_DIR, "log.csv")
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "image_file", "verdict_html", "explanation_html"])
        writer.writerow([ts, img_path, verdict_html_value, explanation_html_value])
    gr.Info("Flagged - saved to the flagged/ folder.")


with gr.Blocks(title="SignalScope") as demo:
    gr.Markdown("# SignalScope - Real vs AI-Generated Detector")
    gr.Markdown(
        "Upload an image to get a likelihood verdict plus a visual explanation "
        "of which regions influenced the decision.",
        elem_classes=["subtitle"],
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Image")
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            verdict_output = gr.HTML(value=VERDICT_PLACEHOLDER, elem_classes=["result-box"])
            heatmap_output = gr.Image(label="Grad-CAM explanation heat-map")
            explanation_output = gr.HTML(value=EXPLANATION_PLACEHOLDER, elem_classes=["explanation-box"])
            flag_btn = gr.Button("Flag")

    submit_btn.click(
        fn=run,
        inputs=image_input,
        outputs=[verdict_output, heatmap_output, explanation_output],
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, verdict_output, heatmap_output, explanation_output],
    )
    flag_btn.click(
        fn=flag_current,
        inputs=[image_input, verdict_output, explanation_output],
        outputs=None,
    )

if __name__ == "__main__":
    demo.launch(share=args.share, css=CSS)
