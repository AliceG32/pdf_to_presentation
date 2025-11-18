import clip
import torch
from PIL import Image

# Загрузка модели
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def classify_table_vs_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    text_descriptions = [
        "a data table with rows and columns",
        "a spreadsheet with numerical data",
        "a structured table with information",
        "a photograph or picture",
        "an image or illustration",
        "a graphic or visual content"
    ]

    text = clip.tokenize(text_descriptions).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    table_prob = sum(probs[0][:3])
    image_prob = sum(probs[0][3:])

    return {
        'is_table': table_prob > image_prob,
        'table_confidence': table_prob,
        'image_confidence': image_prob,
        'all_probabilities': probs[0]
    }

path_to_image = "diploma_text_blocks_sorted_2/page_19_block_001_table.png"
result = classify_table_vs_image(path_to_image)
print(f"Это таблица: {result['is_table']}")
print(f"Уверенность в таблице: {result['table_confidence']:.3f}")
print(f"Уверенность в изображении: {result['image_confidence']:.3f}")
