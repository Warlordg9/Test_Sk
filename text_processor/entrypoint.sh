tr '[:upper:]' '[:lower:]' < "$1" | grep -oP '\w+' | sort | uniq -c | sort -nr
