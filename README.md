s# Prune Wisely, Reconstruct Sharply

Compact 3D Gaussian Splatting training with adaptive pruning and
Difference-of-Gaussian (DoG) rendering.

The repository contains one training entry point, `train_prune_DoG.py`, and a
combined rasterization package with separate CUDA kernels for standard and DoG
rendering.

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

## Dataset

Use a COLMAP scene with this structure:

```text
scene/
  images/
  sparse/0/
```

## Training

```bash
python train_prune_DoG.py -s /path/to/scene --eval
```

Specify an output directory when required:

```bash
python train_prune_DoG.py -s /path/to/scene -m output/scene --eval
```

Training uses standard Gaussian rendering before the pruning stage ends and
automatically switches to DoG rendering afterward. The default run saves the
model at iteration 40,000.

## Rendering

Render the test cameras and save only rendered images and ground truth:

```bash
python render_DoG.py -m output/scene --iteration 40000 --skip_train
```

Results are written to:

```text
output/scene/test/ours_40000/renders/
output/scene/test/ours_40000/gt/
```

## Metrics

```bash
python metrics.py -m output/scene
```

This reports SSIM, PSNR, and LPIPS.

## Full Evaluation

```bash
python full_eval.py \
  -m360 /path/to/mipnerf360 \
  -tat /path/to/tanks_and_temples \
  -db /path/to/deep_blending
```

Use `--skip_training`, `--skip_rendering`, or `--skip_metrics` to run only part
of the evaluation.

## License

This project is derived from the official 3D Gaussian Splatting implementation
and retains its non-commercial research and evaluation license. See
`LICENSE.md`.
