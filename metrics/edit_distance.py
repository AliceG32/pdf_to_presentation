import Levenshtein


def edit_distance_levenshtein(str1, str2):
    distance = Levenshtein.distance(str1, str2)

    # Детальная информация об операциях
    operations = Levenshtein.editops(str1, str2)

    print(f"Строка 1: '{str1}'")
    print(f"Строка 2: '{str2}'")
    print(f"Edit Distance: {distance}")
    print(f"Операции редактирования ({len(operations)}):")

    for op in operations:
        op_name = op[0]
        pos1 = op[1]
        pos2 = op[2]

        if op_name == 'insert':
            print(f"  Вставка: позиция {pos1} → '{str2[pos2]}'")
        elif op_name == 'delete':
            print(f"  Удаление: позиция {pos1} → '{str1[pos1]}'")
        elif op_name == 'replace':
            print(f"  Замена: позиция {pos1} '{str1[pos1]}' → '{str2[pos2]}'")

    return distance, operations


reference_text = "быстрая коричневая лиса прыгает через ленивую собаку"
hypothesis_text = "быстрая коренная лиса прыгает через ленивую собаку"

distance, ops = edit_distance_levenshtein(hypothesis_text, reference_text)