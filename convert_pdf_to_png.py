from pdf2image import convert_from_path
from dotenv import load_dotenv
import os

load_dotenv()

file_path = os.getenv("FILE_PATH")

output_dir = "test_images"
os.makedirs(output_dir, exist_ok=True)

dpi_list = [100, 200, 300, 400, 500]

for dpi in dpi_list:
    images = convert_from_path(file_path, dpi=dpi)
    for i, image in enumerate(images):
        image_name = f"page_{i + 1}_dpi_{dpi}.png"
        image.save(os.path.join(output_dir, image_name), 'PNG')
    print(f"Converted to DPI: {dpi}")
