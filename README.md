# Transformer Meets Boundary Value Inverse Problems

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
```tex
@article{2022GuoCaoChenTransformer,
  title={Transformer Meets Boundary Value Inverse Problems},
  author={Guo, Ruchi and Cao, Shuhao and Chen, Long},
  journal={arXiv preprint arXiv:2209.14977},
  year={2022}
}
```

## Acknowledgments
This work is supported in part by National Science Foundation grants DMS-1913080, DMS-2012465, and DMS-2136075. No additional revenues are related to this work.