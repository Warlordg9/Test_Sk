#!/bin/bash

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <filename> <output_dir>" >&2
    exit 1
fi

filename="$1"
output_dir="$2"

if [ ! -f "$filename" ]; then
    echo "Error: File '$filename' not found" >&2
    exit 1
fi

mkdir -p "$output_dir"

# Подсчёт слов приводим к нижнему регистру, извлекаем слова, сортируем, считаем
echo "Counting words..."
frequency_table=$(tr '[:upper:]' '[:lower:]' < "$filename" | grep -oE '\w+'\'?\'?'\w*|\w+' | sort | uniq -c | sort -nr)

echo "$frequency_table"

# Обработка топ-10 слов
echo -e "\nCreating files for top 10 words..."
rank=1
echo "$frequency_table" | head -n 10 | while read count word; do
    safe_word=$(echo "$word" | tr -cd '[:alnum:]_')
    if [ -z "$safe_word" ]; then
        safe_word="empty_$rank"
    fi
    touch "${output_dir}/${safe_word}_${rank}"
    echo "Created: ${output_dir}/${safe_word}_${rank}"
    rank=$((rank + 1))
done
