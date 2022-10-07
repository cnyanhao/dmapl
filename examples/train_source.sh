#!/usr/bin/env bash
mkdir -p benchmarks/train_source

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/visda-2017 -d VisDA2017Split -s T -t T -a resnet101 --print-freq 1000 --epochs 20 --seed 2022 --center_crop > benchmarks/train_source/VisDA2017_V.txt

# DomainNet
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s c -t c -a resnet101 --epochs 20 --seed 2022 > benchmarks/train_source/DomainNet_c
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s p -t p -a resnet101 --epochs 20 --seed 2022 > benchmarks/train_source/DomainNet_p
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s r -t r -a resnet101 --epochs 20 --seed 2022 > benchmarks/train_source/DomainNet_r
CUDA_VISIBLE_DEVICES=0 python examples/train_source.py data/domainnet -d DomainNet -s s -t s -a resnet101 --epochs 20 --seed 2022 > benchmarks/train_source/DomainNet_s
