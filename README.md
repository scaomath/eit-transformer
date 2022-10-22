# Transformer Meets Boundary Value Inverse Problems
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.10-blue.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2209.14977-b31b1b.svg)](https://arxiv.org/abs/2209.14977)

## Summary
A 2D attention operator is modified according to the integral operator formulation. The modified U-Net drop-in replacement is then used to solve an inverse problem (Electrical Impedance Tomography or EIT). The neural net is used to approximate the inclusion map using a single (or a few) current-to-voltage (Neumann-to-Dirichlet) data pairs. The boundary measurements are preprocessed using a PDE-based feature map.


## Training
Training model: `--model` args can be `uit` (integral transformer), `ut` (with traditional softmax normalization), `hut` (hybrid ut with linear attention), `xut` (cross-attention with hadamard product interaction), `fno2d` (Fourier neural operator 2d), `unet` (traditional UNet with CNN, big baseline, 33m params), `unets` (UNet with the same number of layers with U-integral transformer)

All different models' settings can be found in `configs.yml`.
Default is to train a single input-channel
```bash
python run_train.py --model uit --parts 2 4 5 6
```

## Evaluation
```bash
python evaluation.py --model uit # base integral transformer
python evaluation.py --model uit-c3 --channels 3 # 3 channels
```

## Reference
```bibtex
@article{2022GuoCaoChenTransformer,
  title={Transformer Meets Boundary Value Inverse Problems},
  author={Guo, Ruchi and Cao, Shuhao and Chen, Long},
  journal={arXiv preprint arXiv:2209.14977},
  year={2022}
}
```

## Acknowledgments
This work is supported in part by National Science Foundation grants DMS-1913080, DMS-2012465, and DMS-2136075. No additional revenues are related to this work.