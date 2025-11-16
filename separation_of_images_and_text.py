import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def detect_text_vs_image(image_path):

    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Текстовые промпты для классификации
    text_descriptions = [
        "this is text or writing",
        "this is an image or photograph",
        "this contains printed text",
        "this is a natural scene",
        "this is a graphic design"
    ]

    text_inputs = clip.tokenize(text_descriptions).to(device)

    # Получение предсказаний
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
        probs = logits_per_image.cpu().numpy()[0]

    text_prob = probs[0] + probs[2]
    image_prob = probs[1] + probs[3] + probs[4]

    return {
        'is_text': text_prob > image_prob,
        'text_confidence': text_prob,
        'image_confidence': image_prob,
        'all_probabilities': dict(zip(text_descriptions, probs))
    }


image_path = "diploma/page_44_dpi_100.png"
i2 = "img.png"
# Использование
result = detect_text_vs_image(i2)
print(f"Это текст: {result['is_text']}")
print(f"Уверенность в тексте: {result['text_confidence']:.3f}")
print(f"Уверенность в изображении: {result['image_confidence']:.3f}")
