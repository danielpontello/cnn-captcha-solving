#!/bin/sh

for run in {1..100}
do
    echo "rodando" && python -m memory_profiler test_one.py | grep 60 >> ram_results.txt
done