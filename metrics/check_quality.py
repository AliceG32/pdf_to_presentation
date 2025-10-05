import pytesseract
from PIL import Image
import cv2
import numpy as np
from metrics.wer import calculate_wer_jiwer

def read_txt_file_simple(file_path):
    """
    Простое чтение всего содержимого TXT файла
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Файл {file_path} не найден")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None


image = Image.open("../text_blocks_sorted/block_009.png")

# Использование
reference_text = read_txt_file_simple("../extracted_texts/block_009_page_7.txt")
#reference_text = "быстрая коричневая лиса прыгает через ленивую собаку"
hypothesis_text = pytesseract.image_to_string(image, lang='rus+eng')

result = calculate_wer_jiwer(reference_text, hypothesis_text)
print(f"WER: {result['wer']:.4f} ({result['wer_percentage']:.2f}%)")