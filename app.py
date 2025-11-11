# ------------------------------
# AI Multi‑Modal Assistant — Enhanced Version
# ------------------------------

import gradio as gr
from transformers import pipeline
from PIL import Image
import torch
from torchvision import models, transforms
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import yake  # keyword extraction
import tempfile

# ------------------------------
# 1. Load Models & Labels
# ------------------------------

# NLP pipelines
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)


# Summarization Model
summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn")

# Image classification model
image_model = models.resnet50(pretrained=True)
image_model.eval()
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load ImageNet class labels mapping
imagenet_labels = []
with open("imagenet_classes.txt", "r") as f:  # ensure this file is in your folder
    imagenet_labels = [s.strip() for s in f.readlines()]

# Keyword extraction
kw_extractor = yake.KeywordExtractor(lan="en", top=5)

# ------------------------------
# 2. Helper Functions
# ------------------------------


def analyze_text(text: str) -> dict:
    sentiment = sentiment_model(text)[0]
    summary = summarizer_model(
        text, max_length=min(len(text.split()) + 10, 50), min_length=5
    )[0]["summary_text"]
    keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
    return {
        "Sentiment": sentiment["label"],
        "Sentiment Score": round(sentiment["score"], 3),
        "Summary": summary,
        "Keywords": keywords,
    }


def analyze_image(image: Image.Image) -> dict:
    img_t = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = image_model(img_t)
        class_idx = outputs.argmax().item()
    if 0 <= class_idx < len(imagenet_labels):
        class_label = imagenet_labels[class_idx]
    else:
        class_label = f"Class index {class_idx}"
    return {"Predicted Class Index": class_idx, "Predicted Class Label": class_label}


def generate_pdf(results: dict) -> str:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "AI Multi-Modal Assistant Report")

    y = 720
    for key, value in results.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20
        if y < 60:
            c.showPage()
            y = 750

    c.save()
    buffer.seek(0)

    # ✅ Save to a temp file and return path instead of buffer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(buffer.getvalue())
        tmp_path = tmp.name

    return tmp_path  # ✅ returns file path (Gradio-friendly)


# ------------------------------
# 3. Multi‑Modal Analysis Function
# ------------------------------
def analyze(input_data):
    # Handles both text and image correctly in Gradio
    if isinstance(input_data, str) and input_data.strip():
        return analyze_text(input_data)
    elif isinstance(input_data, dict) and "image" in input_data:
        return analyze_image(input_data["image"])
    elif isinstance(input_data, Image.Image):
        return analyze_image(input_data)
    else:
        return {"Error": "Please enter text or upload an image."}


# ------------------------------
# 4. Gradio UI Layout
# ------------------------------

with gr.Blocks() as demo:
    gr.Markdown("## AI Multi‑Modal Assistant")

    with gr.Tab("Image Analysis"):
        image_input = gr.Image(type="pil", label="Upload an image for classification")
        analyze_image_button = gr.Button("Analyze Image")  # ✅ MOVED INSIDE the tab

        image_output = gr.JSON(label="Image Analysis Results")
        pdf_button_image = gr.Button("Download Report (PDF)")

        analyze_image_button.click(fn=analyze, inputs=image_input, outputs=image_output)

    pdf_button_image.click(
        fn=lambda x: generate_pdf(analyze(x)),
        inputs=image_input,
        outputs=gr.File(label="Download PDF Report"),
    )


    with gr.Tab("Text Analysis"):
        text_input = gr.Textbox(
            label="Enter text to analyze",
            placeholder="Type your text here...",
            lines=5
        )

        analyze_text_button = gr.Button("Analyze Text")
        text_output = gr.JSON(label="Text Analysis Results")
        pdf_button_text = gr.Button("Download Report (PDF)")

        # Text analysis events
        
        analyze_text_button.click(
            fn=analyze,
            inputs=text_input,
            outputs=text_output
        )

        pdf_button_text.click(
            fn=lambda x: generate_pdf(analyze(x)),
            inputs=text_input,
            outputs=gr.File(label="Download PDF Report")
        )


# ------------------------------
# 5. Launch the App
# ------------------------------
demo.launch(share=True)
