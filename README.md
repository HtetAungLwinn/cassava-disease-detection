# Cassava Leaf Disease Detection

This project trains a simple 5-class CNN for cassava leaf disease classification in PyTorch. The current workflow lives in `ex1.ipynb` and expects the Kaggle competition dataset to be downloaded into the project root.

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

Then open `ex1.ipynb`.

## Current notebook behavior

The root notebook currently:

- reads `cassava-leaf-disease-classification/train.csv`
- reads `cassava-leaf-disease-classification/label_num_to_disease_map.json`
- loads images from `cassava-leaf-disease-classification/train_images`
- trains a simple CNN with `nn.CrossEntropyLoss()`

Once the dataset is in place, the next modeling step can be to replace that criterion with focal loss for the class imbalance experiments you wanted to start.
