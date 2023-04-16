set -e
[[ -e posts.csv ]] || curl -OL https://huggingface.co/lodestones/SD-1.5-Test/resolve/main/posts.csv
[[ -e e6_dump ]] || mkdir e6_dump
[[ $(mountpoint -q e6_dump) ]] || sudo mount -t tmpfs -o size=200G tmpfs e6_dump
xargs -n 1 curl -L < files.txt | tar xzf - -C e6_dump/