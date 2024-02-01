This is the code for the submitted paper "Propagation Structure-aware Graph Transformer for Robust and Interpretable Fake News Detection"
## Installation

We have tested our code on `Python 3.10` with `PyTorch 1.13.1`, `PyG 2.2.0` and `CUDA 11.7`. 


Please ensure that the environment in which you run the code is a Linux system equipped with a GPU.

## Dataset
Since the dataset is too large, you should download the dataset from a public source released by previous works。

Specifically, for raw datasets, several earlier studies have already published the raw data at the following location: https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR?usp=sharing

For instance, after downloading the raw data, one can place the data in the `data/politifact/raw directory`.


## Run Examples
```
cd gnn_model
python PSGT
```
