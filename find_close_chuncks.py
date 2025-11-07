import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity  # правильный импорт


def find_most_similar_with_text(query_embedding_path, embeddings_folder, chunks_folder, top_k=3):
    """
    Находит самые близкие чанки и показывает их текст
    """
    query_embedding = np.load(query_embedding_path)
    query_embedding = query_embedding.reshape(1, -1)

    results = []

    for file_name in os.listdir(embeddings_folder):
        if file_name.endswith('.npy') and file_name != os.path.basename(query_embedding_path):
            file_path = os.path.join(embeddings_folder, file_name)
            candidate_embedding = np.load(file_path).reshape(1, -1)

            similarity = cosine_similarity(query_embedding, candidate_embedding)[0][0]

            # Загружаем соответствующий текст
            text_file = file_name.replace('.npy', '.txt')
            text_path = os.path.join(chunks_folder, text_file)

            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()  # первые 200 символов

            results.append({
                'filename': file_name,
                'similarity': similarity,
                'text_preview': text_content
            })

    # Сортируем по убыванию близости
    results.sort(key=lambda x: x['similarity'], reverse=True)

    return results[:top_k]


# Использование
most_similar = find_most_similar_with_text(
    query_embedding_path="diploma_plan_presentation_embeddings_better_frida/plan_diploma_presentation_chunk_007.npy",
    embeddings_folder="diploma_embeddings_frida",
    chunks_folder="diploma_chunks_tokenizer",
    top_k=3
)

print("Самые близкие чанки:")
for i, result in enumerate(most_similar, 1):
    print(f"\n{i}. {result['filename']} (сходство: {result['similarity']:.4f})")
    print(f"Текст: {result['text_preview']}")