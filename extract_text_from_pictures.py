import os
import shutil

import pytesseract
from PIL import Image


def list_files_os(path, output_dir="diploma_extracted_text_tesseract3"):
    """Проходит по всем файлам в папке используя os.listdir()"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        print(file_path)
        if os.path.isfile(file_path):  # Проверяем, что это файл, а не папка
            img = Image.open(file_path)

            text = pytesseract.image_to_string(img, lang='rus+eng')

            txt_filename = f'{str(file_name).split(".")[0]}.txt'
            output_txt_path = os.path.join(output_dir, txt_filename)
            # Сохраняем в файл
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"Текст успешно сохранен в: {output_txt_path}")
        if "table" in file_name:
            output_png_path = os.path.join(output_dir, file_name)
            shutil.copy2(file_path, output_png_path)
            print(f"PNG файл сохранен: {output_png_path}")

folder_with_pictures = "diploma_text_blocks_sorted_3"
list_files_os(folder_with_pictures)


