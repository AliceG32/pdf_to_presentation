from sentence_transformers import SentenceTransformer
import numpy as np
#from langchain.text_splitter import RecursiveCharacterTextSplitter

import re
import os

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Используем мощную русскоязычную модель FRIDA
model = SentenceTransformer('ai-forever/FRIDA')

# Создаем папку для эмбеддингов
path = folder_with_recognized_pages = "diploma_plan_presentation_chunks_better"
output_embeddings_dir = "diploma_plan_presentation_embeddings_better_frida"
os.makedirs(output_embeddings_dir, exist_ok=True)

for file_name in sorted(os.listdir(folder_with_recognized_pages)):
    file_path = os.path.join(path, file_name)

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Очистка текста
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', text)

    print(f"📄 Обрабатывается: {file_name}")
    print(f"📝 Длина текста: {len(cleaned_text)} символов")

    embedding = model.encode(cleaned_text)

    output_file_name = file_name.split(".")[0] + ".npy"
    output_file_path = os.path.join(output_embeddings_dir, output_file_name)

    np.save(output_file_path, embedding)
    print(f"✅ Эмбеддинг сохранен: {output_file_name}")
    print(f"📊 Размер эмбеддинга: {embedding.shape}")
    print(f"🔢 Пример первых 5 значений: {embedding[:5]}")
    print("-" * 50)