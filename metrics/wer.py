import jiwer


def calculate_wer_jiwer(reference, hypothesis):
    # Вычисляем WER
    wer_score = jiwer.wer(reference, hypothesis)

    return {
        'wer': wer_score,
        'wer_percentage': wer_score * 100,
    }


reference_text = "быстрая коричневая лиса прыгает через ленивую собаку"
hypothesis_text = "быстрая коренная лиса прыгает через ленивую собаку"

result = calculate_wer_jiwer(reference_text, hypothesis_text)
print(f"WER: {result['wer']:.4f} ({result['wer_percentage']:.2f}%)")