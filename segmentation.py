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


def is_table_block(image_block, contour_coords, original_image=None, min_lines=3, line_density_threshold=0.1):
    """
    Определяет, является ли блок таблицей на основе нескольких признаков
    """
    table_indicators = 0
    horizontal_lines = detect_horizontal_lines(image_block, min_lines)
    if horizontal_lines >= min_lines:
        table_indicators += 1

    vertical_lines = detect_vertical_lines(image_block, min_lines)
    if vertical_lines >= min_lines:
        table_indicators += 1

    # Считаем блок таблицей если выполнено более половины признаков
    return table_indicators == 2, table_indicators


def detect_horizontal_lines(image, min_line_length=30):
    """Обнаруживает горизонтальные линии со сбалансированными параметрами"""
    height, width = image.shape[:2]

    # Адаптивные параметры
    adaptive_min_length = max(min_line_length, width * 0.2)  # снизили до 20%

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Используем несколько методов для надежности
    # Метод 1: Морфологические операции
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_horizontal, iterations=1)

    # Метод 2: HoughLinesP как резерв
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=adaptive_min_length, maxLineGap=10)

    horizontal_count_morph = 0
    horizontal_count_hough = 0

    # Подсчет линий из морфологического метода
    contours, _ = cv2.findContours(horizontal_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length_ratio = w / width
        aspect_ratio = w / max(h, 1)

        if (length_ratio > 0.3 and  # снизили до 30%
                aspect_ratio > 8 and  # снизили до 8
                w > adaptive_min_length):
            horizontal_count_morph += 1

    # Подсчет линий из Hough метода
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if (angle < 10 or angle > 170) and length > adaptive_min_length:
                horizontal_count_hough += 1

    # Используем максимальное значение из двух методов
    horizontal_count = max(horizontal_count_morph, horizontal_count_hough)

    return horizontal_count


def detect_vertical_lines(image, min_line_length=20):
    """Обнаруживает вертикальные линии со сбалансированными параметрами"""
    height, width = image.shape[:2]

    adaptive_min_length = max(min_line_length, height * 0.2)  # снизили до 20%

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Метод 1: Морфологические операции
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical, iterations=1)

    # Метод 2: HoughLinesP
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=adaptive_min_length, maxLineGap=10)

    vertical_count_morph = 0
    vertical_count_hough = 0

    # Подсчет из морфологического метода
    contours, _ = cv2.findContours(vertical_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length_ratio = h / height
        aspect_ratio = h / max(w, 1)

        if (length_ratio > 0.3 and  # снизили до 30%
                aspect_ratio > 8 and  # снизили до 8
                h > adaptive_min_length):
            vertical_count_morph += 1

    # Подсчет из Hough метода
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if 80 < angle < 100 and length > adaptive_min_length:
                vertical_count_hough += 1

    vertical_count = max(vertical_count_morph, vertical_count_hough)

    return vertical_count



def extract_text_from_pdf_coordinates(pdf_path, coordinates_list, page_num=0, output_dir="diploma_extracted_texts"):
    print(f"Обрабатывается страница: {page_num}")

    print(pdf_path)
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1] if page_num > 0 else doc[0]

    extracted_texts = []

    for i, coords in enumerate(coordinates_list):
        x, y, w, h = coords
        rect = fitz.Rect(x, y, x + w, y + h)

        text = page.get_text("text", clip=rect).strip()

        filename = f"block_{i + 1:03d}_page_{page_num + 1}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        extracted_texts.append({
            'block_id': i + 1,
            'coordinates': coords,
            'text': text,
            'rect': rect,
            'filename': filename
        })

        print(f"Блок {i + 1}: {text[:100]}...")  # Показываем только первые 100 символов

    doc.close()
    return extracted_texts


def segment_text_blocks(image_path, page_num=0):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=3)

    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    text_blocks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > 1000:
            text_blocks.append((x, y, w, h))

    text_blocks = sorted(text_blocks, key=lambda block: (block[1], block[0]))

    result = img.copy()

    # Анализируем каждый блок на предмет таблицы
    table_blocks = []
    regular_blocks = []

    for i, (x, y, w, h) in enumerate(text_blocks):
        block_roi = img[y:y + h, x:x + w]

        # Проверяем, является ли блок таблицей
        is_table, confidence = is_table_block(block_roi, (x, y, w, h), img)

        if is_table:
            color = (0, 0, 255)  # Красный для таблиц
            table_blocks.append((x, y, w, h))
            block_type = "TABLE"
        else:
            color = (0, 255, 0)  # Зеленый для обычного текста
            regular_blocks.append((x, y, w, h))
            block_type = "TEXT"

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, f'{i + 1}({block_type})', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(f"Всего блоков: {len(text_blocks)}")
    print(f"Таблиц: {len(table_blocks)}")
    print(f"Текстовых блоков: {len(regular_blocks)}")

    extract_text_from_pdf_coordinates(
        file_path,
        text_blocks, page_num)

    return result, text_blocks, img, table_blocks


def extract_text_blocks_sorted(image_path, output_dir="diploma_text_blocks_sorted_2", page_num=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_image, blocks, original_img, table_blocks = segment_text_blocks(image_path, page_num)

    extracted_blocks = []

    for i, (x, y, w, h) in enumerate(blocks):
        block_roi = original_img[y:y + h, x:x + w]

        # Определяем тип блока
        is_table, confidence = is_table_block(block_roi, (x, y, w, h), original_img)
        block_type = "table" if is_table else "text"

        block_filename = os.path.join(output_dir, f"page_{page_num}_block_{i + 1:03d}_{block_type}.png")
        cv2.imwrite(block_filename, block_roi)

        extracted_blocks.append({
            'id': i + 1,
            'coordinates': (x, y, w, h),
            'filename': block_filename,
            'image': block_roi,
            'type': block_type,
            'confidence': confidence
        })

        print(f"Блок {i + 1} ({block_type}): координаты ({x}, {y}, {w}, {h})")

    return result_image, extracted_blocks

doc = fitz.open(file_path)
page_count = doc.page_count

for i in range(1, page_count):
    num_page = i
    image_path = f"diploma/page_{num_page}_dpi_100.png"
    result_image, blocks_info = extract_text_blocks_sorted(image_path, page_num=num_page)

    # Визуализация результатов
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

    # Подсчет типов блоков
    table_count = sum(1 for block in blocks_info if block['type'] == 'table')
    text_count = sum(1 for block in blocks_info if block['type'] == 'text')

    plt.title(f"Найдено блоков: {len(blocks_info)} (Таблиц: {table_count}, Текст: {text_count})")
    plt.axis('off')
    plt.tight_layout()
    #plt.show()

    cv2.imwrite("segmented_result_sorted_with_tables.jpg", result_image)

    print("\n" + "=" * 50)
    print("ОБНАРУЖЕННЫЕ ТАБЛИЦЫ:")
    print("=" * 50)
    for block in blocks_info:
        if block['type'] == 'table':
            print(f"Блок {block['id']}: confidence={block['confidence']}/5")
            print(f"Координаты: {block['coordinates']}")
            print(f"Файл: {block['filename']}")
            print("-" * 30)