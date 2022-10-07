#!/usr/bin/env bash
mkdir -p benchmarks/dmapl_visda2017split
mkdir -p benchmarks/dmapl_domainnet

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/visda-2017 -d VisDA2017Split -s T -t V -a resnet101 --print-freq 1000 --epochs 30 --seed 2022 --center_crop --prob_th 0.9 --param_sce 0.01 > benchmarks/dmapl_visda2017split/VisDA2017Split_resnet101_sce001_th09.txt

# DomainNet
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s c -t p -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_c2p.txt
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s c -t r -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_c2r.txt
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s c -t s -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_c2s.txt

CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s p -t c -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_p2c.txt
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s p -t r -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_p2r.txt
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s p -t s -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_p2s.txt

CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s r -t c -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_r2c.txt
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s r -t p -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_r2p.txt
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s r -t s -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_r2s.txt

CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s s -t c -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_s2c.txt
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s s -t p -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_s2p.txt
CUDA_VISIBLE_DEVICES=0 python examples/dmapl.py data/domainnet -d DomainNet -s s -t r -a resnet101 -p 500 --epochs 30 --seed 2022 > benchmarks/dmapl_domainnet/DomainNet_s2r.txt
