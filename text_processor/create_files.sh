mkdir -p "$2"
words=$(tr '[:upper:]' '[:lower:]' < "$1" | grep -oP '\w+' | sort | uniq -c | sort -nr | head -10 | awk '{print $2}')
counter=1
for word in $words; do 
  touch "$2/${word}_$((counter++))"
done
