# Prune Wisely, Reconstruct Sharply

Official implementation of the CVPR 2026 paper:

**CVPR 2026**

[[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Prune_Wisely_Reconstruct_Sharply_Compact_3D_Gaussian_Splatting_via_Adaptive_CVPR_2026_paper.html)]
[[arXiv](https://arxiv.org/abs/2602.24136)]

Compact 3D Gaussian Splatting training with adaptive pruning and
Difference-of-Gaussian (DoG) rendering.

## Requirements

- Windows or Linux
- NVIDIA GPU
- CUDA toolkit compatible with PyTorch 1.12.1
- Conda

## Installation

Clone the repository with its submodules:

```bash
git clone --recursive https://github.com/WangHaoran16/Prune-Wisely-Reconstruct-Sharply.git
cd Prune-Wisely-Reconstruct-Sharply
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate gaussian_splatting
```

If the CUDA rasterizer needs to be rebuilt after a source change:

```bash
pip install submodules/dog-gaussian-rasterization --force-reinstall --no-deps
```

## Training

```bash
python train_prune_DoG.py -s /path/to/scene --eval
```

Specify an output directory when required:

```bash
python train_prune_DoG.py -s /path/to/scene -m output/scene --eval
```

## Rendering

```bash
python render_DoG.py -m output/scene
```

## Metrics

```bash
python metrics.py -m output/scene
```

## Full Evaluation

```bash
python full_eval.py \
  -m360 /path/to/mipnerf360 \
  -tat /path/to/tanks_and_temples \
  -db /path/to/deep_blending
```

## Bibtex

```bibtex
@inproceedings{wang2026prune,
  title={Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives},
  author={Wang, Haoran and Huang, Guoxi and Zhang, Fan and Bull, David and Anantrasirichai, Nantheera},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11716--11725},
  year={2026}
}
```

## License

This project is derived from the official 3D Gaussian Splatting implementation
and retains its non-commercial research and evaluation license. See
`LICENSE.md`.
