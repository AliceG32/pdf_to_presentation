import cv2
import fitz
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from collections import Counter
from dotenv import load_dotenv
import os

load_dotenv()

file_path = os.getenv("FILE_PATH")


def segment_page_final(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return

    original_height, original_width = img.shape[:2]

    # Конвертация в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Улучшение контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Инвертирование для работы с белым фоном
    binary = 255 - binary

    # Удаление шума
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    tables = is_table_block(binary)

    images = detect_images_final(img, binary)

    text_blocks = detect_text_blocks_comprehensive(binary, tables, images)

    all_blocks = []

    for block in text_blocks:
        x, y, w, h = block
        all_blocks.append({'type': 'text', 'bbox': (x, y, w, h), 'coords': (x, y)})

    for block in tables:
        x, y, w, h = block
        all_blocks.append({'type': 'table', 'bbox': (x, y, w, h), 'coords': (x, y)})

    for block in images:
        x, y, w, h = block
        all_blocks.append({'type': 'image', 'bbox': (x, y, w, h), 'coords': (x, y)})

    sorted_blocks = sort_all_blocks_by_coordinates(all_blocks)

    visualize_results_detailed(img, sorted_blocks, binary)

    return sorted_blocks


def sort_all_blocks_by_coordinates(all_blocks):
    """Сортировка всех блоков по координатам: сначала по Y, затем по X"""
    if not all_blocks:
        return []

    sorted_blocks = sorted(all_blocks, key=lambda b: (b['coords'][1], b['coords'][0]))
    return sorted_blocks


def detect_text_blocks_comprehensive(binary, tables, images):
    """Комплексное обнаружение текстовых блоков с несколькими методами"""
    # Создание маски для исключения таблиц и изображений
    mask = np.ones(binary.shape, dtype=np.uint8) * 255

    for (x, y, w, h) in tables + images:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)

    text_only = cv2.bitwise_and(binary, binary, mask=mask)

    blocks1 = morphological_text_detection_final(text_only)

    blocks2 = connected_components_text_detection(text_only)

    blocks3 = projection_based_detection(text_only)

    all_blocks = blocks1 + blocks2 + blocks3

    merged_blocks = merge_similar_blocks(all_blocks)

    filtered_blocks = filter_text_blocks(merged_blocks, binary.shape)

    return filtered_blocks


def morphological_text_detection_final(binary):
    """Морфологическое обнаружение текста с улучшенными параметрами"""
    text_blocks = []

    # Пробуем разные размеры ядер для разных типов текста
    kernels = [
        (60, 3),  # Для широких строк текста
        (40, 8),  # Для обычных абзацев
        (25, 15),  # Для высоких блоков
        (80, 5),  # Для очень широких блоков
    ]

    for kw, kh in kernels:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))

        grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if (w >= 5 and h >= 5 and
                    w < 0.98 * binary.shape[1] and h < 0.98 * binary.shape[0]):

                # плотность текста в блоке
                roi = binary[y:y + h, x:x + w]
                if roi.size > 0:
                    text_density = np.sum(roi) / (255 * roi.size)
                    if text_density > 0.03:
                        text_blocks.append((x, y, w, h))

    return text_blocks


def connected_components_text_detection(binary):
    """Обнаружение текста через Connected Components с мягкими параметрами"""
    text_blocks = []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if (area >= 50 and
                w >= 15 and h >= 8 and
                w < 0.95 * binary.shape[1] and h < 0.95 * binary.shape[0] and
                w / h < 20 and h / w < 20):
            text_blocks.append((x, y, w, h))

    return text_blocks


def projection_based_detection(binary):
    """Проекционный анализ для обнаружения пропущенных текстовых блоков"""
    text_blocks = []

    horizontal_proj = np.sum(binary, axis=1)
    h_threshold = np.max(horizontal_proj) * 0.02

    in_text = False
    text_start = 0

    for i, value in enumerate(horizontal_proj):
        if value > h_threshold and not in_text:
            in_text = True
            text_start = i
        elif value <= h_threshold and in_text:
            in_text = False
            text_end = i
            height = text_end - text_start

            if height >= 8:

                line_region = binary[text_start:text_end, :]
                vertical_proj = np.sum(line_region, axis=0)
                v_threshold = np.max(vertical_proj) * 0.05

                in_column = False
                col_start = 0

                for j, v_value in enumerate(vertical_proj):
                    if v_value > v_threshold and not in_column:
                        in_column = True
                        col_start = j
                    elif v_value <= v_threshold and in_column:
                        in_column = False
                        col_end = j
                        width = col_end - col_start

                        if width >= 25:

                            block_region = line_region[:, col_start:col_end]
                            density = np.sum(block_region) / (255 * block_region.size)
                            if density > 0.02:
                                text_blocks.append((col_start, text_start, width, height))

    return text_blocks


def merge_similar_blocks(blocks, overlap_threshold=0.4):
    """Объединение похожих и перекрывающихся блоков"""
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    merged = []
    current = list(blocks[0])

    for block in blocks[1:]:
        x1, y1, w1, h1 = current
        x2, y2, w2, h2 = block

        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        overlap_area = x_overlap * y_overlap
        min_area = min(w1 * h1, w2 * h2)

        if (overlap_area > min_area * overlap_threshold or
                (abs(y1 - y2) < max(h1, h2) * 0.5 and
                 abs((x1 + w1 / 2) - (x2 + w2 / 2)) < max(w1, w2) * 0.8)):

            # Объединяем блоки
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y
            current = [new_x, new_y, new_w, new_h]
        else:
            merged.append(tuple(current))
            current = list(block)

    merged.append(tuple(current))
    return merged


def filter_text_blocks(blocks, image_shape):
    """Фильтрация текстовых блоков по разумным критериям"""
    filtered = []
    img_height, img_width = image_shape

    for x, y, w, h in blocks:
        # Основные критерии
        if (w >= 5 and h >= 5 and  # Минимальные размеры
                w <= img_width * 0.95 and h <= img_height * 0.95 and
                (w * h) >= 200 and (w * h) <= img_width * img_height * 0.7):

            aspect_ratio = w / h
            if 0.1 < aspect_ratio < 15:
                filtered.append((x, y, w, h))

    return filtered


def is_table_block(binary):
    """Обнаружение таблиц"""
    tables = []

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Минимальная площадь таблицы
            x, y, w, h = cv2.boundingRect(contour)
            if w < 0.9 * binary.shape[1] and h < 0.9 * binary.shape[0]:
                tables.append((x, y, w, h))

    return tables


def detect_images_final(original_img, binary):
    """Обнаружение изображений"""
    images = []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 1500 < area < binary.shape[0] * binary.shape[1] * 0.6:
            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 4:
                roi = original_img[y:y + h, x:x + w]
                if is_likely_image_final(roi):
                    images.append((x, y, w, h))

    return images


def is_likely_image_final(roi):
    """Проверка, является ли область изображением"""
    if roi.size == 0:
        return False

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    std_dev = np.std(gray_roi)

    return entropy > 3.5 and std_dev > 15


def visualize_results_detailed(img, all_blocks, binary):
    """Детальная визуализация результатов с объединенными блоками"""
    result_img = img.copy()
    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    text_count = 0
    table_count = 0
    image_count = 0

    for i, block in enumerate(all_blocks):
        block_type = block['type']
        x, y, w, h = block['bbox']

        if block_type == 'text':
            color = (0, 255, 0)
            label = f'T{text_count}'
            text_count += 1
        elif block_type == 'table':
            color = (255, 0, 0)
            label = f'Table{table_count}'
            table_count += 1
        elif block_type == 'image':
            color = (0, 0, 255)
            label = f'Img{image_count}'
            image_count += 1
        else:
            color = (255, 255, 0)
            label = f'Unknown{i}'

        thickness = 3 if block_type in ['table', 'image'] else 2
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, thickness)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, thickness)

        cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Отображаем результаты
    """fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f'Результаты сегментации (всего блоков: {len(all_blocks)})')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Бинарное изображение с выделенными блоками')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()"""

    print("\n=== ОБЩИЙ ПОРЯДОК ВСЕХ БЛОКОВ ===")
    for i, block in enumerate(all_blocks):
        x, y, w, h = block['bbox']
        print(f"  {i:2d}. {block['type']:6} ({x:4d}, {y:4d}) - {w:4d}x{h:3d}")


def debug_missed_text(image_path):
    """Отладка: поиск пропущенных текстовых областей"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    methods = [
        ('OTSU', cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ('Adaptive Mean', cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 11, 2)),
        ('Adaptive Gaussian', cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Оригинал')
    axes[0, 0].axis('off')

    for idx, (name, binary) in enumerate(methods, 1):
        binary = 255 - binary
        row, col = idx // 2, idx % 2

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_img = img.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 10 and w * h > 200:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        axes[row, col].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'{name} - Найдено: {len(contours)}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def extract_all_blocks_sorted(image_path, page_num=0, output_dir="diploma_text_blocks_sorted_3"):
    """
    Извлекает и сохраняет все типы блоков (текст, таблицы, изображения) с сортировкой по координатам

    Args:
        image_path (str): Путь к исходному изображению
        output_dir (str): Директория для сохранения блоков
        page_num (int): Номер страницы

    Returns:
        tuple: (result_image, extracted_blocks)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_blocks = segment_page_final(image_path)

    extracted_blocks = []

    for i, block in enumerate(all_blocks):
        block_type = block['type']
        x, y, w, h = block['bbox']

        block_roi = cv2.imread(image_path)
        if block_roi is not None:
            block_roi = block_roi[y:y + h, x:x + w]
        else:
            print(f"Ошибка загрузки изображения для блока {i}")
            continue

        block_filename = os.path.join(output_dir, f"page_{page_num}_block_{i + 1:03d}_{block_type}.png")
        cv2.imwrite(block_filename, block_roi)

        extracted_blocks.append({
            'id': i + 1,
            'type': block_type,
            'coordinates': (x, y, w, h),
            'filename': block_filename,
            'image': block_roi,
            'area': w * h
        })

        print(f"Блок {i + 1} ({block_type}): координаты ({x}, {y}, {w}, {h}), площадь: {w * h} пикселей")

    return extracted_blocks


doc = fitz.open(file_path)
page_count = doc.page_count

for i in range(1, page_count):
    num_page = i
    image_path = f"diploma/page_{num_page}_dpi_100.png"
    # image_path = "job_offer_pages_png/page_5_dpi_100.png"
    # image_path =  "art_pages_png/page_10_dpi_100.png"

    all_blocks = extract_all_blocks_sorted(image_path, i)
