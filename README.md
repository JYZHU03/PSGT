This is the code for the paper "Propagation Structure-aware Graph Transformer for Robust and Interpretable Fake News Detection"
## Installation

We have tested our code on Linux system equipped with `Python 3.10` with `PyTorch 1.13.1`, `PyG 2.2.0` and `CUDA 11.7`. 

## Dataset
Since the dataset is too large, one can download the dataset from a public source released by previous works.

Specifically, for raw datasets, the earlier study have already published the raw data at the following location: https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR?usp=sharing

For instance, after downloading the raw data, one can place the data in the `data/politifact/raw` directory.


## Run Examples
```
cd gnn_model
python PSGT
```

## Citation

If you find this work useful, please cite our KDD 2024 paper:
```bibtex
@inproceedings{zhu2024propagation,
  title={Propagation Structure-Aware Graph Transformer for Robust and Interpretable Fake News Detection},
  author={Zhu, Junyou and Gao, Chao and Yin, Ze and Li, Xianghua and Kurths, Juergen},
  booktitle={Proceedings of the 30th ACM SIGKDD international conference on knowledge discovery \& data mining},
  year={2024}
}
```
