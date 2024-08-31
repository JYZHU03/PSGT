The implementation for the paper "Propagation Structure-aware Graph Transformer for Robust and Interpretable Fake News Detection"
## Installation

We have tested our code on a Linux system equipped with `Python 3.10`, `PyTorch 1.13.1`, `PyG 2.2.0` and `CUDA 11.7`.

## Dataset
Since the dataset is too large, we provide the raw and processed data in: https://drive.google.com/drive/folders/1ZoCzpBcl5UIdmhKV1Q2eo6kzi5Ch_rz1?usp=sharing

After downloading the data, one can place it in the `data` directory.


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
  pages={4652--4663},
  year={2024}
}
```
