FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y gawk sed grep coreutils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY word_count.sh /app/word_count.sh
RUN chmod +x /app/word_count.sh

ENTRYPOINT ["/app/word_count.sh"]
