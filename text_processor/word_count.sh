#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <filename> <output_dir>"
    exit 1
fi

filename=$1
output_dir=$2

if [ ! -f "$filename" ]; then
    echo "Error: File $filename not found!"
    exit 1
fi

mkdir -p "$output_dir"

echo "Counting word frequencies:"
cat "$filename" | \
    tr '[:upper:]' '[:lower:]' | \
    tr -cs '[:alpha:]' '\n' | \
    sort | \
    uniq -c | \
    sort -nr

top_words=$(cat "$filename" | \
    tr '[:upper:]' '[:lower:]' | \
    tr -cs '[:alpha:]' '\n' | \
    sort | \
    uniq -c | \
    sort -nr | \
    head -10 | \
    awk '{print $2}')

count=1
for word in $top_words; do
    touch "${output_dir}/${word}_${count}"
    echo "Created file: ${output_dir}/${word}_${count}"
    ((count++))
done
