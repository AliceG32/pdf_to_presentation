import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import re


def sort_key(filename):
    numbers = re.findall(r'page_(\d+)_chunk_(\d+)', filename)
    if numbers:
        return (int(numbers[0][0]), int(numbers[0][1]))
    return (0, 0)


def merge_texts_to_chunks(input_folder):
    chunks = []  # Список для хранения текстовых фрагментов
    chunk_files = []  # Список для хранения имен исходных файлов

    if not os.path.exists(input_folder):
        print(f"Папка '{input_folder}' не найдена!")
        return chunks, chunk_files

    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    if not txt_files:
        print("Нет txt-файлов для обработки!")
        return chunks, chunk_files

    sorted_files = sorted(txt_files, key=sort_key)

    for filename in sorted_files:
        file_path = os.path.join(input_folder, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read().strip()
                if content:
                    chunks.append(content)
                    chunk_files.append(filename)  # Сохраняем имя файла
        except:
            try:
                with open(file_path, 'r', encoding='cp1251') as infile:
                    content = infile.read().strip()
                    if content:
                        chunks.append(content)
                        chunk_files.append(filename)
            except:
                print(f"Пропущен файл: {filename}")

    print(f"Обработано {len(chunks)} текстовых фрагментов")
    return chunks, chunk_files


def save_top_results(results, indices, chunk_files, page_num, output_folder="search_results"):
    """
    Сохраняет найденные текстовые фрагменты в отдельные файлы
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Создана папка '{output_folder}'")

    saved_files = []

    for i, ((text, score), idx) in enumerate(zip(results, indices)):
        # Получаем имя исходного файла чанка
        original_filename = chunk_files[idx]
        # Убираем расширение .txt
        base_name = os.path.splitext(original_filename)[0]
        # Создаем имя для результата
        result_filename = f"page_{int(page_num):03d}_{base_name}_score_{score:.3f}.txt"
        result_path = os.path.join(output_folder, result_filename)

        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"Источник: {original_filename}\n")
                f.write(f"Схожесть: {score:.4f}\n")
                f.write("=" * 50 + "\n")
                f.write(text)

            saved_files.append(result_filename)
            print(f"Сохранен: {result_filename}")

        except Exception as e:
            print(f"Ошибка при сохранении файла {result_filename}: {e}")

    return saved_files


# Загружаем чанки и сохраняем имена файлов
chunks, chunk_files = merge_texts_to_chunks("diploma_chunks")

model = SentenceTransformer("intfloat/multilingual-e5-base")
embs = model.encode(chunks, normalize_embeddings=True).astype("float32")

# Индекс на скалярное произведение (IP) — эквивалент косинусу при нормализованных векторах
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)


def faiss_top_n(query: str, n: int = 3):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    sims, idx = index.search(q, n)
    out = []
    indices_list = []
    for score, i in zip(sims[0], idx[0]):
        out.append((chunks[int(i)], float(score)))
        indices_list.append(int(i))
    return out, indices_list


path = "diploma_plan_presentation_chunks"
# Выполняем поиск
for file_name in os.listdir(path):
    print("-"*50)
    page_num = file_name.split(".")[0].split("_")[-1]
    file_path = os.path.join(path, file_name)
    with open(file_path, 'r', encoding='utf-8') as infile:
        content = infile.read().strip()
        query = content
        results, indices = faiss_top_n(query, n=3)

        print("Результаты поиска:")
        for i, (text, score) in enumerate(results):
            original_file = chunk_files[indices[i]]
            print(f"{i + 1}. Схожесть: {score:.3f} | Источник: {original_file}")
            print(f"   Текст: {text}...")
            print()
        saved_files = save_top_results(results, indices, chunk_files, page_num)
