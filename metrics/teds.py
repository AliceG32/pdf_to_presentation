import csv

import pandas as pd
from table_recognition_metric import TEDS


def csv_to_html_table(csv_path, title="Table"):

    df = pd.read_csv(csv_path)
    table_html = df.to_html(index=False)

    full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
        </head>
        <body>
            <h1>{title}</h1>
            {table_html}
        </body>
        </html>
      """

    return full_html


def calculate_teds_csv(gt_csv, pred_csv):
    gt_html = csv_to_html_table(gt_csv)
    pred_html = csv_to_html_table(pred_csv)


    print(gt_html)
    print(pred_html)
    teds = TEDS()  # structure_only=True для сравнения только структуры

    score = teds(gt_html, pred_html)
    return score


def html_file_to_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_string = file.read()
    return html_string

def calculate_teds_html(reference_file_path, hypothesis_file_path):
    reference_table = html_file_to_string(reference_file_path)
    hypothesis_table = html_file_to_string(hypothesis_file_path)

    teds = TEDS()

    score = teds(reference_table, hypothesis_table)
    return score


if __name__ == "__main__":
    table2 = "table_1_1.csv"
    table1 = "table_1_2.csv"

    score = calculate_teds_csv(table1, table2)
    print(f"TEDS score: {score:.4f}")