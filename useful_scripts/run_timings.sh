#!/bin/bash

batch_sizes=(2 4 8 16 32 64 128 256 512)

for batch_size in "${batch_sizes[@]}"; do
	echo "Running timing (no profiler scaffoling) with batch size $batch_size..."
  python useful_scripts/obtain_dataloader.py --batch_size "$batch_size" --timer
done

echo "All runs completed."
