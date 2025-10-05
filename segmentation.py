import cv2
from matplotlib import pyplot as plt
import os


def segment_text_blocks(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 5)) # параметры ядра надо получше подбирать
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
    for i, (x, y, w, h) in enumerate(text_blocks):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(result, f'{i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result, text_blocks, img


def extract_text_blocks_sorted(image_path, output_dir="text_blocks_sorted"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_image, blocks, original_img = segment_text_blocks(image_path)

    extracted_blocks = []

    for i, (x, y, w, h) in enumerate(blocks):

        block_roi = original_img[y:y + h, x:x + w]

        block_filename = os.path.join(output_dir, f"block_{i + 1:03d}.png")
        cv2.imwrite(block_filename, block_roi)

        extracted_blocks.append({
            'id': i + 1,  # Нумерация с 1
            'coordinates': (x, y, w, h),
            'filename': block_filename,
            'image': block_roi
        })

        print(f"Блок {i + 1}: координаты ({x}, {y}, {w}, {h}), сохранен как {block_filename}")

    return result_image, extracted_blocks


result_image, blocks_info = extract_text_blocks_sorted("test_images/page_5_dpi_100.png")

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title(f"Найдено текстовых блоков: {len(blocks_info)} (нумерация сверху вниз, слева направо)")
plt.axis('off')
plt.tight_layout()
plt.show()

cv2.imwrite("segmented_result_sorted.jpg", result_image)

