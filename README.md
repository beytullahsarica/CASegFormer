# CASegFormer: What Matters in Transformer Decoders for Pixel-Level Crack Segmentation?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

This repository contains the code for **CASegFormer** and for the controlled study in our paper. You will find
the model, the nine baselines we re-trained under the same protocol on Crack500, DeepCrack and CFD, the switches
for every ablation, and the scripts that produce the tables, figures and statistical tests.

> **Paper:** *What Matters in Transformer Decoders for Pixel-Level Crack Segmentation? A Controlled Study with
> a Crack-Aware SegFormer.* Beytullah Sarica. Under review.

## Why this study

Cracks are thin, often only one to three pixels wide, and they are easy to confuse with stains and shadows.
Many crack networks add new modules to the decoder, yet the reported gains are often smaller than the
differences between evaluation protocols. We wanted to know which parts of a transformer segmenter really
help, so every model here is trained and evaluated in exactly the same way.

## The model

CASegFormer keeps the Mix Transformer encoder (MiT-B2) of SegFormer and replaces its decoder. The new decoder
works from coarse to fine: it upsamples step by step, fuses the skip feature at every scale, and predicts at
the full input resolution. On top of this path sit four crack-specific modules and three structural components.

| Component | What it does |
|---|---|
| **CSCA**, cross-scale crack attention | lets each feature scale attend to its neighbours |
| **CFE**, crack frequency enhancement | a learnable gate on the Fourier amplitude of the features |
| **AFF**, adaptive feature fusion | gates the decoder and skip features instead of plain concatenation |
| **EGBAM**, edge-guided boundary attention | a supervised edge map and deformable convolution at crack borders |
| **FRR**, full-resolution recovery | a shallow image branch at full and half resolution for hairline cracks |
| **OGP**, orientation-guided propagation | predicts the crack direction and refines the prediction along it |
| **Phase-field head** | predicts an Ambrosio–Tortorelli phase field for each crack (used for training) |

The loss combines focal, Dice, clDice, deep-supervision and edge terms with three terms from phase-field
fracture: a phase-field target, crack-surface energy and orientation consistency.

## What we found

- CASegFormer has the highest mean F1 of the ten re-trained models on all three datasets. Its lead over
  SegFormer with the same encoder ranges from 0.7 F1 points on Crack500 (1896 training images) to 7.3 points
  on CFD (82 training images).
- On the two smaller datasets, most of this lead comes from decoding progressively to full resolution. A plain
  U-Net decoder on the same MiT-B2 encoder recovers much of it, and CASegFormer stays 0.7 points above that
  decoder on Crack500 and 2.1 points on CFD.
- The four decoder modules can be removed while the structural components are kept. Mean F1 drops by 0.05
  to 0.48 points and the computation falls from 48 to 22 GFLOPs. The structural components, in turn, do not
  change F1 at 256×256 input when the modules are kept, but they give crack orientation and phase-field maps.
- Test-time augmentation and seed ensembling move F1 as much as these architectural choices, so we report
  them separately.

All models share the data pipeline, the schedule and the data loss terms. Checkpoints and binarisation
thresholds are always chosen on the **validation set**, never on the test set.

## Results

Test F1 (%) at 256×256 input, single model, no test-time augmentation, threshold chosen on the validation
set. Values with ± are the mean and standard deviation over three seeds; the others are from one seed.

| Model | Encoder | Params (M) | GFLOPs | Crack500 | DeepCrack | CFD |
|---|---|---:|---:|---:|---:|---:|
| LinkNet | ResNet-34 | 21.8 | 10.1 | 73.03 | 83.20 | 65.65 |
| U-Net | ResNet-34 | 24.4 | 15.6 | 73.32 | 82.21 | 67.07 |
| UNet++ | ResNet-34 | 26.1 | 36.8 | 73.27 | 84.01 ± 0.53 | 68.69 |
| DeepLabV3+ | ResNet-34 | 22.4 | 15.8 | 72.28 | 81.29 | 61.97 |
| SegFormer | MiT-B2 | 24.7 | 11.1 | 73.70 ± 0.23 | 81.60 ± 0.70 | 62.87 ± 0.22 |
| U-Net | MiT-B2 | 27.5 | 14.2 | 73.75 ± 0.13 | 83.65 ± 1.23 | 68.10 ± 0.34 |
| UPerNet | ConvNeXt-T | 37.0 | 38.2 | 73.58 ± 0.23 | 82.72 ± 0.29 | 65.02 ± 0.42 |
| DeepCrack† | VGG16-BN | 29.5 | 84.8 | 73.32 | 83.23 | 67.54 |
| HrSegNet-B48† | – | 6.5 | 4.2 | 72.43 | 81.87 | 64.99 |
| **CASegFormer** | MiT-B2 | 33.0 | 48.0 | **74.41 ± 0.06** | **84.92 ± 0.71** | **70.17 ± 0.25** |
| CASegFormer without the four modules | MiT-B2 | 27.5 | 22.1 | 74.37 ± 0.42 | 84.44 ± 0.23 | 70.07 ± 0.55 |

† Our re-implementation, following the original paper.

The paper also reports precision, recall, IoU, mIoU, F1 with a 2-pixel tolerance and clDice, together with
per-image Wilcoxon tests, confidence intervals over seeds, bootstrap intervals (on Crack500 also by source
photograph), all ablations, cross-dataset transfer and a crack-width analysis. With four-flip test-time
augmentation and an ensemble of the three seeds, CASegFormer reaches 74.84% F1 on Crack500.

## What is in the repository

```
train.py                  training, model selection on the validation set and testing, one command per run
models/                   CASegFormer (models/casegformer.py), the model registry and two re-implemented baselines
losses/                   data loss terms and the phase-field loss terms
data/                     loaders and augmentation for Crack500, DeepCrack and CFD
utils/                    training loop, inference (TTA, CC filter), metrics, complexity, reports, figures
splits/                   our CFD split (82 / 15 / 21 images)
tools/
  split_cfd.py            recreates the CFD split from the original dataset
  ensemble_eval.py        averages the probability maps of several trained runs
  benchmark_speed.py      measures the FPS of all models in one run
  compute_aiu.py          AIU, the metric of the original Crack500 paper
  crack_width_analysis.py recall per crack-width bin
  make_internal_maps.py   edge map, phase field, orientation and before/after-OGP maps of a trained model
  make_paper_tables.py    the LaTeX tables of the paper, from results/*/test_results*.json
  seed_stats.py           confidence intervals, bootstrap and per-photograph tests, fixed-threshold F1
  make_paper_figures.py   the paper figures, from the saved results
```

## Installation

```bash
git clone https://github.com/beytullahsarica/CASegFormer.git
cd CASegFormer
conda create -n casegformer python=3.10 -y
conda activate casegformer
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

We trained on a single NVIDIA V100 (16 GB). The ImageNet-pretrained encoder weights are downloaded on the
first run, so that run needs internet access.

## Data

The datasets are not included here. Please download them from their authors:

| Dataset | Source | Split we use |
|---|---|---|
| Crack500 | [Yang et al., FPHBN](https://github.com/fyangneil/pavement-crack-detection) | the original one: 1896 train / 348 val / 1124 test |
| DeepCrack | [Liu et al., DeepCrack](https://github.com/yhlleo/DeepCrack) | 300 train / 237 test; 10% of train is held out for validation (`--split_seed 42`) |
| CFD | [Shi et al., CrackForest](https://github.com/cuilimeng/CrackForest-dataset) | our split: 82 train / 15 val / 21 test, lists in `splits/` |

CFD comes without an official split. `tools/split_cfd.py` recreates ours: it sorts the images into five bins
by their fraction of crack pixels and divides each bin 70/15/15 with seed 42. The resulting lists are in
`splits/cfd_{train,val,test}.txt`.

```bash
python tools/split_cfd.py --input_dir cfd_original --output_dir ../datasets/CFD
```

Arrange the data like this. An image and its mask share the file name, and mask pixels above 127 are crack.

```
datasets/
├── Crack500/{train,val,test}/{images,masks}/
├── DeepCrack/{train,test}/{images,masks}/     # no val/ folder: 10% of train is held out automatically
└── CFD/{train,val,test}/{images,masks}/
```

The code looks for `../datasets` next to the repository. If your data is elsewhere, pass
`--data_root /path/to/datasets`.

## Training and evaluation

One command trains a model, keeps the best checkpoint on the validation set, picks the threshold on the
validation set and tests it. It also runs two extra tests with test-time augmentation (four flips, and eight
views) plus connected-component filtering. The results go to
`results/<dataset>_<model>_<backbone>_<size>_<tag>_s<seed>/`.

```bash
# CASegFormer on Crack500, with the settings of the paper (we used seeds 0, 1 and 2)
python train.py --dataset crack500 --model casegformer --backbone mit_b2 --tag pgl3 --seed 0 \
    --epochs 120 --batch_size 8 --img_size 256

# DeepCrack and CFD are small, so they use CutMix and longer schedules (300 and 1000 epochs)
python train.py --dataset deepcrack --model casegformer --backbone mit_b2 --tag pgl3 --seed 0 \
    --epochs 300 --batch_size 8 --img_size 256 --use_cutmix
python train.py --dataset cfd --model casegformer --backbone mit_b2 --tag pgl3 --seed 0 \
    --epochs 1000 --batch_size 8 --img_size 256 --use_cutmix

# A baseline uses only the data loss terms and the same schedule as CASegFormer on that dataset
python train.py --dataset crack500 --model upernet --backbone tu-convnext_tiny --no_physics --seed 0 \
    --epochs 120 --batch_size 8 --img_size 256
```

The baselines (`--model` / `--backbone`) are `linknet`, `unet`, `unetplusplus` and `deeplabv3plus` with
`resnet34`; `segformer` with `mit_b2`; `unet` with `mit_b2` (a plain U-Net decoder on the encoder of
CASegFormer); `upernet` with `tu-convnext_tiny`; `deepcrack` with `vgg16`; and `hrsegnet` with `none`. Add
`--no_pretrained` for HrSegNet, which is trained from scratch as in its paper. Baselines always use
`--no_physics`.

**Ablations.** Each component has a switch: `--no_csca`, `--no_cfe`, `--no_aff` (concatenation instead),
`--no_egbam`, `--no_detail` (FRR), `--no_ogp`, `--no_pf_head`, `--no_physics` (data loss terms only) and
`--lambda_side 0` (no deep supervision). Give each variant its own `--tag`. For example, the decoder without
its four modules:

```bash
python train.py --dataset cfd --model casegformer --backbone mit_b2 --tag pgl3-nomodules --seed 0 \
    --no_csca --no_cfe --no_aff --no_egbam \
    --epochs 1000 --batch_size 8 --img_size 256 --use_cutmix
```

**Original image size.** Use `--eval_size 0` to train on 256×256 crops at the original scale and test on
whole images, instead of resizing everything to 256×256.

**Evaluating a finished run again**, for example with test-time augmentation or on another dataset:

```bash
python train.py --eval_dir results/<experiment> --tta d4 --use_cc_filter
python train.py --eval_dir results/<experiment> --test_dataset cfd          # cross-dataset test
```

**Seed ensemble:**

```bash
python tools/ensemble_eval.py --out_dir results/crack500_casegformer_ensemble3 \
    --runs results/crack500_casegformer_mitb2_256_pgl3_s0 results/crack500_casegformer_mitb2_256_pgl3_s1 \
           results/crack500_casegformer_mitb2_256_pgl3_s2
```

If a run is interrupted, `--auto_resume` continues it from the last checkpoint. On a SLURM cluster, wrap each
command in a job script with your own `#SBATCH` header.

### What a run writes

| File | Content |
|---|---|
| `best_model.pth` | the best weights (raw or EMA) and the validation threshold |
| `config.json` | all arguments and the data split |
| `test_results.json` | precision, recall, F1, IoU, mIoU, F1 with 2-px tolerance, clDice, ODS/OIS, per-image mean ± std |
| `test_results_tta-flip_cc.json`, `test_results_tta-d4_cc.json` | the same with test-time augmentation and CC filter |
| `test_per_image.csv` | metrics for every test image (used for the Wilcoxon tests) |
| `complexity.json` | parameters, GFLOPs, FPS |
| `probs/` | test probability maps (8-bit PNG) |
| `fig_*.png`, `experiment_report.md` | training curves, PR curve, sample predictions and a full report |

## Reproducing the tables and figures

Once the runs are finished:

```bash
python tools/benchmark_speed.py --dataset crack500   # FPS of all models on one GPU
for DS in crack500 deepcrack cfd; do python tools/crack_width_analysis.py --dataset $DS; done
python tools/make_paper_tables.py                    # LaTeX tables (mean ± std over seeds, Wilcoxon tests)
python tools/seed_stats.py                           # confidence intervals, bootstrap, per-photograph tests
python tools/make_paper_figures.py --dataset crack500 # paper figures (PNG)
for DS in deepcrack cfd; do
    python tools/make_paper_figures.py --dataset $DS --qual_models unetplusplus segformer upernet
done
python tools/compute_aiu.py --dataset crack500       # AIU of every run
python tools/make_internal_maps.py --exp_dir results/crack500_casegformer_mitb2_256_pgl3_s0
```

## Trained models

The nine CASegFormer models from the paper (three seeds on each dataset) are on Zenodo:
**[DOI: 10.5281/zenodo.24437075](https://doi.org/10.5281/zenodo.24437075)**. Each model comes with its training
configuration, its validation threshold and its test scores. Copy a model folder into `results/` and run
`python train.py --eval_dir results/<folder>` to evaluate it.

| Model | Dataset | Test F1 (%), mean of 3 seeds |
|---|---|---:|
| CASegFormer (MiT-B2) | Crack500 | 74.41 ± 0.06 |
| CASegFormer (MiT-B2) | DeepCrack | 84.92 ± 0.71 |
| CASegFormer (MiT-B2) | CFD | 70.17 ± 0.25 |

The weights are for research use only (see [License](#license)).

## Citation

If this code or the models help your work, please cite:

```bibtex
@article{sarica2026casegformer,
  title   = {What Matters in Transformer Decoders for Pixel-Level Crack Segmentation? A Controlled Study with a Crack-Aware SegFormer},
  author  = {Sarica, Beytullah},
  journal = {Under review},
  year    = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The datasets keep the licenses of their authors. The
trained weights start from the ImageNet-pretrained MiT encoder of [SegFormer](https://github.com/NVlabs/SegFormer)
(NVIDIA Source Code License, non-commercial research use) and were trained on Crack500, DeepCrack and CFD, so
they are for research use only.

## Acknowledgements

The encoders and several baselines come from
[segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) and
[timm](https://github.com/huggingface/pytorch-image-models), and the MiT encoder follows
[SegFormer](https://github.com/NVlabs/SegFormer). Many thanks to the authors of
[Crack500](https://github.com/fyangneil/pavement-crack-detection),
[DeepCrack](https://github.com/yhlleo/DeepCrack) and
[CrackForest](https://github.com/cuilimeng/CrackForest-dataset) for sharing their data.

## Contact

Beytullah Sarica, Istanbul Technical University
([ORCID 0000-0002-6660-8508](https://orcid.org/0000-0002-6660-8508)). For questions or bugs, please open a
GitHub issue.
