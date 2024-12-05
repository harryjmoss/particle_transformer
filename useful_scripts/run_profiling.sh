#!/bin/bash

batch_sizes=(2 4 8 16 32 64 128 256 512)

for batch_size in "${batch_sizes[@]}"; do
  echo "Running profiling with batch size $batch_size..."
  python obtain_dataloader.py --batch_size "$batch_size"
done

echo "All runs completed."
