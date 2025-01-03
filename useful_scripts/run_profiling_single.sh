#!/bin/bash

batch_sizes=(2) 
for batch_size in "${batch_sizes[@]}"; do
  echo "Running profiling with batch size $batch_size..."
  python useful_scripts/profiling_steerer.py --batch_size "$batch_size" --backward
done

echo "All runs completed."
