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

## Run the suggested training loop with the suggested settings

```sh
./train_JetClass.sh ParT full
```

