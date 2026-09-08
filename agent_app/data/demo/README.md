# MENET demo datasets

Run `python scripts/prepare_demo_datasets.py` to reproduce the converted CSV files from pinned public R package data.

## Rice

The rice datasets are derived from `Rice_Zhao_etal` in RAINBOWR. They contain Rice Diversity Panel marker genotypes and field phenotypes described by Zhao et al. (2010, 2011). The converter creates plant-height and Arkansas flowering-time MENET datasets. RAINBOWR is MIT licensed; retain the original scientific citations when publishing results.

Source: https://github.com/KosukeHamazaki/RAINBOWR

## Wheat

The wheat datasets are derived from `wheat.X` and `wheat.Y` in BGLR. They contain 599 CIMMYT wheat lines, 1,279 binary DArT markers, and standardized grain yield in four environments. The converter creates MENET datasets for environments 1 and 2. BGLR is GPL-3 licensed; the data source is CIMMYT.

Source: https://github.com/gdlc/BGLR-R

These datasets are intended for software testing and method demonstration. Their inclusion does not make model importance equivalent to GWAS significance or biological causality.
