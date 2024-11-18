# Full installation & running instructions for particle_transformer

[Particle Transformer](https://github.com/jet-universe/particle_transformer) runs with a [weaver backend](https://github.com/hqucms/weaver-core), and is in reality a wrapper for weaver. The main purpose of Particle Transformer is seemingly to provide a simple interface for running weaver with a set of default options that are useful for particle physics analyses.

These instructions refer specifically to running the standard particle_transformer-recommended training of the `ParticleTransformer` network on the `JetClass` dataset, _specifically_ on an Ada compute instance available via STFC. The compute instance is a virtual machine (VM) created with an NVIDIA A100 GPU, which is used by the weaver backend for model training.

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

This installs `torch` with support for CUDA 11.8, which is the version of CUDA available on the Ada VMs. If you try to install torch via `pip install torch`, you won't be able to access GPU devices when running the code as this will install a version of torch that supports CUDA >=12.0.

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

The command to run `particle_transformer` suggested by the repository is:

```sh
./train_JetClass.sh ParT full
```

The contents of this script (note that no defaults are overridden here) are:

```sh
#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=50
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((10000 * 128))
dataopts="--num-workers 2 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# currently only Pythia
SAMPLE_TYPE=Pythia

$CMD \
    --data-train \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/train_100M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/train_100M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/train_100M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/train_100M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/train_100M/ZJetsToNuNu_*.root" \
    --data-val "${DATADIR}/${SAMPLE_TYPE}/val_5M/*.root" \
    --data-test \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/test_20M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/test_20M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/test_20M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/test_20M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/test_20M/ZJetsToNuNu_*.root" \
    --data-config data/JetClass/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log logs/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}${suffix} \
    "${@:3}"
```

Skipping over the first portion of this script (we don't really care where the data directory is for example), there are some sections of particular interest.

- We're only using a single GPU here. If there are multiple GPUs available (and we've passed this via `$DDP_NGPUS`) then some distributed training is handled with `torchrun`. This still calls the `weaver` backend.
- Defaults are chosen for epochs, samples per epoch, and samples per epoch at the validation stage.
  - Samples per epoch is calculated based on the number of GPUs available, as a default.
  - The number of epochs is set to 50
  - `--num-workers` is set to 2, so uses two processes to run the command
  - `--fetch-step` is set to 0.01. An attempt to explain this is made in the weaver repository, but in effect this represents the percentage of samples that are used from each of the input data files. This is a way to reduce the amount of data that is read from disk, and is useful when the input data is very large. Ultimately you should set a smaller percentage if you encounter out of memory CUDA errors when running the code. In practice, the process will just be killed and won't throw a useful error message.
- We've specified `ParT` as the model, so the portion of the script dealing with that model is invoked.
    - Learning rate starts at `1e-3`
    - batch size is set to 512

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

## Backend code and model architecture

Please see [weaver_code_review.md](./weaver_code_review.md) for an overview of the underlying neural network training framework and the ParticleTransformer model architecture implementation in code.
