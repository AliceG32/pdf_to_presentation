import pdfplumber
from pathlib import Path
import camelot
from dotenv import load_dotenv
import os

load_dotenv()

file_path = os.getenv("FILE_PATH")


def process_pdf_pages_separately(pdf_path, output_dir="pdf_pages_text"):
    """
    Обрабатывает каждую страницу PDF отдельно и сохраняет результаты
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = Path(pdf_path).stem

    results = {
        'pdf_info': {},
        'pages': []
    }

    with pdfplumber.open(pdf_path) as pdf:
        results['pdf_info'] = {
            'filename': Path(pdf_path).name,
            'total_pages': len(pdf.pages),
        }

        for page_num, page in enumerate(pdf.pages):
            print(f"Обрабатывается страница {page_num + 1}...")

            page_data = process_single_page(page, page_num + 1, pdf_name, output_dir, pdf_path)
            results['pages'].append(page_data)

    return results


def process_single_page(page, page_number, pdf_name, output_dir, pdf_path):
    """
    Обрабатывает одну страницу и сохраняет все данные
    """
    page_text = page.extract_text() or ""
    words = page.extract_words()
    tables = camelot.read_pdf(pdf_path, flavor="stream", pages="all")
    print(tables[0].parsing_report)

    page_data = {
        'page_number': page_number,
        'text': page_text,
        'tables': tables,
        'files': {}
    }

    page_data['files'] = save_page_data(page_data, page_number, pdf_name, output_dir)

    return page_data


def save_page_data(page_data, page_number, pdf_name, output_dir):
    """
    Сохраняет данные страницы в различные форматы
    """
    files_created = {}
    page_prefix = f"{pdf_name}_page_{page_number:03d}"

    txt_path = os.path.join(output_dir, f"{page_prefix}_text.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(page_data['text'])
    files_created['text'] = txt_path

    if page_data['tables']:
        tables1 = page_data['tables']
        for i, table in enumerate(tables1):
            print(type(table))
            table_csv_path = os.path.join(output_dir, f"{page_prefix}_table_{i + 1:02d}.csv")
            filename = f"{page_prefix}_table_{i + 1}.csv"
            table.df.to_csv(table_csv_path, index=False, encoding='utf-8-sig')
            files_created[f'table_{i + 1}_csv'] = table_csv_path
            print(f"Сохранено: {filename}")

    return files_created


process_pdf_pages_separately(file_path)
