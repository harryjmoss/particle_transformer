# Particle Transformer runbook

This details the process for running `particle_transformer` from scratch on an Ada VM.

## Clone the repository from github

```sh
git clone https://github.com/harryjmoss/particle_transformer.git
cd particle_transformer
```

This uses my ([@harryjmoss](https://github.com/harryjmoss)) fork of Particle Transformer.

## Create a virtual environment

```sh
conda create -n your-environment-name python=3.11
conda activate your-environment-name
```

## Install dependencies via pip

```
pip install weaver-core requests
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Note the non-standard `torch` installation command, as Ada has CUDA 11.8 installed rather than versions 12.X.
Installing a version of pytorch compiled against the correct version of CUDA saves on any installation of newer CUDA versions.

## Download the JetClass dataset

```sh
./get_datasets.py JetClass -d datasets
```

Downloads the JetClass dataset to `./datasets`

The entire dataset is downloaded from zenodo and is available at [this link](https://zenodo.org/records/6619768)

Dataset is NOT gzipped, but is tarballed. Total size 189.8 GB

On an Ada VM, downloading at roughly 110 MiB/s.

- The training data consists of 1000 `.root` files, all from 107-201 MB (probably averaging at 150 MB).
- Similarly the Validation data consists of 50 of the same size root files.
- Test data consists of 200 files with the same range of file sizes.

## Run the suggested training loop with the suggested settings

```sh
./train_JetClass.sh ParT full
```

## Performance testing

```sh
time ./train_JetClass.sh ParT full
```

Results:

```sh
real 2210m34.033s
user 3043m28.783s
sys 1715m57.355s
```

N.B.: Real time == wall time
Wall time was 36 hours 50 minutes 34.033 seconds.

On Ada VM with 1x NVIDIA A100 GPU.