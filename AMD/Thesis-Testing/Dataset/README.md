# Dataset Preparation

The full datasets are intentionally excluded from this repository because of
their size. Only configuration and metadata files are retained.

Datasets used:

- CIFAR-10 for ResNet-50
- COCO 2017 MiniTrain 10K for YOLOv8m
- GLUE SST-2 for DistilBERT

The benchmark scripts prepare or download the required assets locally before
testing. Dataset downloads and preprocessing are excluded from benchmark
timing.
