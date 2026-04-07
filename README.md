# Cassava Leaf Disease Detection

This project tracks a series of PyTorch experiments for 5-class cassava leaf disease classification. The notebooks progress from a baseline CNN to several ResNet18 and focal-loss training variants, and they expect the Kaggle competition dataset to be downloaded into the project root. The latest Kaggle submission notebook is configured to run without internet by loading the pretrained ResNet18 checkpoint from a local Kaggle input.

## Experiment change log

- `ex1.ipynb` (`563a16e`, 2026-03-30): baseline 5-class `SimpleCNN` trained with `nn.CrossEntropyLoss()` and standard image resizing.
- `ex2.ipynb` (`8e4e03d`, 2026-04-03): added custom `FocalLoss(gamma=2.0)` while keeping the original `SimpleCNN` pipeline.
- `ex3.ipynb` (`81d0058` and `5aadb40`, 2026-04-03): replaced the baseline CNN with a pretrained `ResNet18` backbone and trained that architecture with focal loss.
- `ex4.ipynb` (`5da4e53`, 2026-04-03): expanded training with class-weighted focal loss, ImageNet normalization, heavier augmentation (`RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`), and a `StepLR` scheduler.
- `ex5.ipynb` (`27e7104`, 2026-04-03): fine-tuned the ResNet18 setup by reducing augmentation to horizontal flips only, changing focal loss to `gamma=1.0`, removing class weights, and lowering the learning rate to `3e-4`.

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

Then open the notebook variant you want to run, starting with `ex1.ipynb` for the baseline or `ex5.ipynb` for the latest ResNet18 experiment.

## 5. Kaggle offline ResNet18 setup

If you want to submit `ex5.ipynb` to the Kaggle competition without enabling internet:

1. Download the official `resnet18-f37072fd.pth` checkpoint once outside the competition rerun.
2. Upload that file as any Kaggle Dataset input.
3. Attach the input to the notebook and manually set `WEIGHTS_PATH` in `ex5.ipynb` to the exact file location under `/kaggle/input/...`.
4. Keep internet disabled in the Kaggle notebook settings and rerun all cells.

## Latest notebook behavior (`ex5.ipynb`)

The latest experiment currently:

- reads `cassava-leaf-disease-classification/train.csv`
- reads `cassava-leaf-disease-classification/label_num_to_disease_map.json`
- loads images from `cassava-leaf-disease-classification/train_images`
- uses a manual `WEIGHTS_PATH` you can edit to point at `resnet18-f37072fd.pth` inside your attached Kaggle Dataset input
- loads a pretrained `ResNet18` checkpoint from that local file instead of downloading weights during execution
- keeps Kaggle internet access disabled for submission-safe reruns
- trains with `FocalLoss(gamma=1.0)`
- uses `Adam` with a learning rate of `3e-4`
- applies resize and ImageNet normalization everywhere, with horizontal flip augmentation only on the training set
- steps the learning rate with `torch.optim.lr_scheduler.StepLR`

All notebooks use the same cassava dataset layout, so you can compare the experiment progression directly across `ex1.ipynb` through `ex5.ipynb`.
