The implementation for the paper "Propagation Structure-aware Graph Transformer for Robust and Interpretable Fake News Detection"
## Installation

We have tested our code on a Linux system equipped with `Python 3.10`, `PyTorch 1.13.1`, `PyG 2.2.0` and `CUDA 11.7`.

## Dataset
Since the dataset is too large, we provide the raw and processed `politifact` data.

One can easily download the raw dataset of `gossipcop` from a public source released by previous works: https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR?usp=sharing

After downloading the raw data, one can place the gossipcop data in the `data/gossipcop/raw` directory.


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
