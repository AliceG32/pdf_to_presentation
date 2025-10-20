import os
import shutil

def combine_texts(path, list_of_file_names, output_dir="diploma_recognized_pages"):
    """Проходит по всем файлам в папке используя os.listdir()"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    text = ""
    page_number = 1
    block_number = 1
    for file_name in list_of_file_names:
        file_path = os.path.join(path, file_name)
        print(file_path)

        if f'page_{page_number}' not in file_path:
            if text != "":
                txt_filename = f'page_{page_number}_block_{block_number:03d}_text.txt'
                output_txt_path = os.path.join(output_dir, txt_filename)

                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                print(f"Текст успешно сохранен в: {output_txt_path}")
            page_number += 1
            block_number = 1
            text = ""
        if "table" not in file_name:
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                text += " "
                text += content

        else:
            if text and text.strip():
                txt_filename = f'page_{page_number}_block_{block_number:03d}_text.txt'
                output_txt_path = os.path.join(output_dir, txt_filename)

                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                print(f"Текст успешно сохранен в: {output_txt_path}")

                block_number += 1

            destination_folder = output_dir
            file_extension = file_name.split(".")[-1]
            new_filename = f'page_{page_number}_block_{block_number:03d}_table.{file_extension}'

            # Создаем папку назначения, если её нет
            os.makedirs(destination_folder, exist_ok=True)

            # Полный путь к новому файлу
            destination_path = os.path.join(destination_folder, new_filename)

            # Копируем файл
            shutil.copy2(file_path, destination_path)

            print(f"PNG файл сохранен: {destination_path}")

            block_number += 1
            text = ""


folder_with_pictures = "diploma_extracted_text_tesseract"
list_of_files_names = []
for file_name in sorted(os.listdir(folder_with_pictures)):
    list_of_files_names.append(file_name)
list_of_files_names = sorted(list_of_files_names, key=lambda x: (int(x.split("_")[1]), int(x.split("_")[3])))

combine_texts(folder_with_pictures, list_of_files_names)
