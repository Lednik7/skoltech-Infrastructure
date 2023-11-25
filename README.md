# skoltech-Infrastructure

## Project Structure

```
├── artifacts
│   ├── images
│   └── weights
├── data
│   ├── digital_leaders
│   │   ├── images  - contains images in .png format
│   │   └── masks  - contains masks in .png format
│   └── external_data
├── .gitignore
├── LICENSE
├── Makefile
├── notebooks
│   ├── EDA.ipynb
│   ├── ensemble_test.ipynb - evaluating ensemble
│   ├── predict_test.ipynb - predicting on test set
│   ├── split_merge_test.ipynb  - splitting and merging check
│   └── torchgeo_baseline.ipynb  - main script for training
├── poetry.lock
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── modelling
    │   ├── __init__.py
    │   ├── base.py
    │   ├── ensemble.py
    │   ├── metrics.py
    │   ├── predict.py
    │   ├── production
    │   │   ├── __init__.py
    │   │   ├── deeplabMOCO.py
    │   │   ├── footPrint.py
    │   │   ├── FPNMOCO.py
    │   │   ├── unetMOCO.py
    │   │   └── unetPlusPlusMOCO.py
    │   └── train.py
    ├── preprocessing
    │   ├── __init__.py
    │   ├── reader.py
    │   └── tile_generating.py  - script to generate tiles from train data
    └── utils
        ├── __init__.py
        └── base.py
```

## Installation

Required Python.

Data installation:

```bash
make install_data
```

Generate tiles for training:

```bash
make generate_tiles
```

Dependencies and environment:

```bash
make setup
```

## Usage

`torchgeo_baseline.ipynb` - train single segmentation model.

## Contact
If something go wrong please contact me:
`tg: @werserk`