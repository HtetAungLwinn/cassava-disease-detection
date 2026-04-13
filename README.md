# Cassava Leaf Disease Detection

This project tracks a series of PyTorch experiments for 5-class cassava leaf disease classification. The notebooks progress from a baseline CNN to several ResNet18 and focal-loss training variants, and they expect the Kaggle competition dataset to be downloaded into the project root. The latest staged fine-tuning notebook is configured to run without internet by loading the pretrained ResNet18 checkpoint from a local Kaggle input.

## Experiment change log

- `ex1.ipynb` (`563a16e`, 2026-03-30): baseline 5-class `SimpleCNN` trained with `nn.CrossEntropyLoss()` and standard image resizing.
- `ex2.ipynb` (`8e4e03d`, 2026-04-03): added custom `FocalLoss(gamma=2.0)` while keeping the original `SimpleCNN` pipeline.
- `ex3.ipynb` (`81d0058` and `5aadb40`, 2026-04-03): replaced the baseline CNN with a pretrained `ResNet18` backbone and trained that architecture with focal loss.
- `ex4.ipynb` (`5da4e53`, 2026-04-03): expanded training with class-weighted focal loss, ImageNet normalization, heavier augmentation (`RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`), and a `StepLR` scheduler.
- `ex5.ipynb` (`27e7104`, 2026-04-03): fine-tuned the ResNet18 setup by reducing augmentation to horizontal flips only, changing focal loss to `gamma=1.0`, removing class weights, and lowering the learning rate to `3e-4`.
- `ex8.ipynb` (local, 2026-04-08): restarts from the `ex5` ResNet18 baseline and adds staged fine-tuning with gradual unfreezing (`fc` -> `layer4` -> `layer3 + layer4`), per-stage learning rates, best-checkpoint restore, and validation tracking.
- `ex9.ipynb` (local, 2026-04-08): keeps the `ex8` offline ResNet18 setup but switches to a stronger schedule with a longer `fc` warmup (`4` epochs at `1e-3`), full-model fine-tuning (`6` epochs at `1e-4`), BatchNorm freezing only during warmup, automatic checkpoint path resolution, and a best validation accuracy of `82.62%` on the saved run.
- `ex10.ipynb` (local, 2026-04-08): keeps the `ex9` schedule but replaces focal loss with `nn.CrossEntropyLoss(label_smoothing=0.1)`, reaching `83.01%` validation accuracy at global epoch `9` in the `full_model` stage. This improved on `ex9.ipynb` (`82.62%`) but still trailed the best `ex5.ipynb` run (`84.23%`).
- `ex11.ipynb` (local, 2026-04-09): returns to the stronger `ex5` full-model training style, keeps `nn.CrossEntropyLoss(label_smoothing=0.1)`, adds best-checkpoint restore, and currently has the best Kaggle leaderboard result among the notebook runs below.

## Kaggle leaderboard results

These scores are from the saved Kaggle submissions mapped to the notebook variants:

| Notebook | Kaggle submission label | Private score | Public score | Notes |
|---|---|---:|---:|---|
| `ex11.ipynb` | `ex10_training - ex11` | `0.8437` | `0.8439` | Current best single-model leaderboard result |
| `ex5.ipynb` | `Training Ex3 - Version 3` | `0.8436` | `0.8428` | Essentially tied with `ex11.ipynb`; best pre-label-smoothing result |
| `ex10.ipynb` | `ex10_training - ex10` | `0.8366` | `0.8317` | Better than `ex8.ipynb`, but below `ex5.ipynb` and `ex11.ipynb` |
| `ex8.ipynb` | `ex8_training - Version 1` | `0.8304` | `0.8329` | First staged fine-tuning attempt |
| `ex7.ipynb` | `Train Resnet34 - Version 4` | `0.7237` | `0.7316` | Not competitive versus the ResNet18 runs |

Based on the current leaderboard, `ex11.ipynb` is the safest single model to use, and `ex5.ipynb` is the strongest backup or ensemble candidate.

## Project layout

Expected structure after setup:

```text
Project/
|-- ex1.ipynb
|-- requirements.txt
|-- README.md
|-- venv/
|-- scripts/
|   `-- download_kaggle_dataset.ps1
`-- cassava-leaf-disease-classification/
    |-- train.csv
    |-- label_num_to_disease_map.json
    `-- train_images/
```

## 1. Create and activate the virtual environment

PowerShell:

```powershell
cd "c:\Users\genso\Documents\College_Projects\CV 2610\Project"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:

- The notebook metadata currently targets Python 3.12.
- If you want a GPU-specific PyTorch build, replace the `torch` and `torchvision` lines with the install command from the official PyTorch selector for your CUDA version.

## 2. Set up Kaggle API credentials

1. Sign in to Kaggle.
2. Open `Account` -> `Settings`.
3. In the `API` section, click `Create New Token`.
4. Save the downloaded `kaggle.json` file to:

```text
C:\Users\<your-username>\.kaggle\kaggle.json
```

5. Lock down the file permissions in PowerShell:

```powershell
icacls "$env:USERPROFILE\.kaggle\kaggle.json" /inheritance:r /grant:r "$env:USERNAME:F"
```

Important:

- You must also join or accept the competition rules in the Kaggle web UI before CLI downloads will work.
- In the current machine state, `C:\Users\genso\.kaggle\` exists, but `kaggle.json` is still missing.

## 3. Download the dataset

After the token is in place and the venv is active, run:

```powershell
.\scripts\download_kaggle_dataset.ps1
```

This script will:

- use the Kaggle CLI inside `venv`
- download the `cassava-leaf-disease-classification` competition archive
- extract it into `.\cassava-leaf-disease-classification\`

If you prefer the raw CLI command, use:

```powershell
.\venv\Scripts\kaggle.exe competitions download -c cassava-leaf-disease-classification -p .\cassava-leaf-disease-classification
Expand-Archive -LiteralPath .\cassava-leaf-disease-classification\cassava-leaf-disease-classification.zip -DestinationPath .\cassava-leaf-disease-classification -Force
```

## 4. Launch the notebook

```powershell
.\venv\Scripts\Activate.ps1
jupyter lab
```

Then open the notebook variant you want to run, starting with `ex1.ipynb` for the baseline, `ex5.ipynb` for the clean ResNet18 restart point, `ex8.ipynb` for the first staged fine-tuning follow-up, `ex9.ipynb` for the stronger warmup plus full-model fine-tuning schedule, or `ex10.ipynb` for the label-smoothed cross-entropy comparison.

## 5. Kaggle offline ResNet18 setup

If you want to submit `ex5.ipynb`, `ex8.ipynb`, `ex9.ipynb`, or `ex10.ipynb` to the Kaggle competition without enabling internet:

1. Download the official `resnet18-f37072fd.pth` checkpoint once outside the competition rerun.
2. Upload that file as any Kaggle Dataset input.
3. Attach the input to the notebook and manually set `WEIGHTS_PATH` in the notebook you are running to the exact file location under `/kaggle/input/...`.
4. Keep internet disabled in the Kaggle notebook settings and rerun all cells.

## Latest notebook behavior (`ex10.ipynb`)

The latest staged fine-tuning experiment currently:

- reads `cassava-leaf-disease-classification/train.csv`
- reads `cassava-leaf-disease-classification/label_num_to_disease_map.json`
- loads images from `cassava-leaf-disease-classification/train_images`
- uses a manual `WEIGHTS_PATH` you can edit to point at `resnet18-f37072fd.pth` inside your attached Kaggle Dataset input
- can also resolve `WEIGHTS_PATH` when you point it at the parent Kaggle dataset directory instead of the exact `.pth` file
- loads a pretrained `ResNet18` checkpoint from that local file instead of downloading weights during execution
- keeps Kaggle internet access disabled for submission-safe reruns
- trains with `nn.CrossEntropyLoss(label_smoothing=0.1)`
- trains in two stages: `head_only` for `4` epochs at `1e-3`, then `full_model` for `6` epochs at `1e-4`
- freezes BatchNorm running statistics only during the head warmup, then lets the full model adapt normally
- uses `StepLR(step_size=2, gamma=0.5)` inside each stage
- applies resize and ImageNet normalization everywhere, with horizontal flip augmentation only on the training set
- tracks validation metrics per stage and restores the best validation checkpoint before inference
- writes `best_resnet18_ex10_best.pth` alongside `submission.csv`
- reached `83.01%` validation accuracy in the saved run at global epoch `9`, improving on `ex9.ipynb` (`82.62%`) and `ex8.ipynb` (`79.02%`) but still below the best `ex5.ipynb` run (`84.23%`)

`ex5.ipynb` remains the clean restart baseline, `ex8.ipynb` is the first layer-wise fine-tuning attempt, `ex9.ipynb` is the stronger follow-up that switches to full-model fine-tuning after warmup, and `ex10.ipynb` isolates the effect of label smoothing on top of that schedule. All notebooks use the same cassava dataset layout, so you can compare the experiment progression directly across `ex1.ipynb` through `ex10.ipynb`.
