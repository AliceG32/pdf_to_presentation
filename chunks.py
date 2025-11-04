import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import os
import re

def sort_key(filename):
    numbers = re.findall(r'page_(\d+)_block_(\d+)_text', filename)
    if numbers:
        return (int(numbers[0][0]), int(numbers[0][1]))
    return (0, 0)


def merge_texts_to_chunks(input_folder):
    chunks = []  # Список для хранения текстовых фрагментов

    if not os.path.exists(input_folder):
        print(f"Папка '{input_folder}' не найдена!")
        return chunks

    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    if not txt_files:
        print("Нет txt-файлов для обработки!")
        return chunks

    sorted_files = sorted(txt_files, key=sort_key)

    for filename in sorted_files:
        file_path = os.path.join(input_folder, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read().strip()
                if content:
                    chunks.append(content)
        except:
            try:
                with open(file_path, 'r', encoding='cp1251') as infile:
                    content = infile.read().strip()
                    if content:
                        chunks.append(content)
            except:
                print(f"Пропущен файл: {filename}")

    print(f"Обработано {len(chunks)} текстовых фрагментов")
    return chunks


chunks = merge_texts_to_chunks("diploma_recognized_pages_2")

model = SentenceTransformer("intfloat/multilingual-e5-base")
embs = model.encode(chunks, normalize_embeddings=True).astype("float32")

# Индекс на скалярное произведение (IP) — эквивалент косинусу при нормализованных векторах
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

def faiss_top_n(query: str, n: int = 3):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    sims, idx = index.search(q, n)
    out = []
    for score, i in zip(sims[0], idx[0]):
        out.append((chunks[int(i)], float(score)))
    return out

results = faiss_top_n("Введение в тему курсового проекта и его актуальность в контексте международного рынка телеком операторов.", n=10)
for text, score in results:
    print(f"{score:.3f}  {text}")
