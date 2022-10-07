## Setup

Install environment
```
pip install -e .
```
Create dataset VisDA2017Split
```
python split_visda2017.py
```
Train source model
```
sh examples/train_source.py
```
Train target model
```
sh examples/dmapl.py
```
