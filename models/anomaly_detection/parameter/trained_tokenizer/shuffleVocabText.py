import random

# ファイルの読み込み
with open('vocab_size_10051/vocab.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 行をランダムに並べ替え
random.shuffle(lines)

# 結果を同じファイルに書き込む
with open('vocab_size_10051/vocab.txt', 'w', encoding='utf-8') as file:
    file.writelines(lines)

print("vocab.txtの行をランダムに並べ替えました。")
