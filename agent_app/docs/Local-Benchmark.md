# MENET local runtime benchmark

Measured on 2026-09-02 with an NVIDIA GeForce RTX 5060 (8,151 MiB), PyTorch 2.13.0+cu130, CUDA, batch size 8, and one FastAPI thread worker. Times cover dataset validation, trait-specific encoder training, relatedness generation, MENET training, and evaluation. Queue waiting is excluded.

| Dataset | Shape | Epochs | Execution | Test R2 |
| --- | ---: | ---: | ---: | ---: |
| Rice plant height | 357 x 1,311 | 50 | 43 s | 0.537 |
| Rice flowering, Arkansas | 349 x 1,311 | 50 | 41 s | 0.315 |
| Wheat yield, environment 1 | 599 x 1,279 | 50 | 74 s | 0.031 |
| Wheat yield, environment 2 | 599 x 1,279 | 50 | 75 s | -0.102 |

These are engineering runtime measurements, not evidence of model validity. R2 varies with phenotype, split, seed, training configuration, and dataset suitability. The benchmark does not include SNP-level explanation, repeated training, bootstrap, permutation tests, concurrent GPU jobs, upload time, or network transfer.

During benchmarking, two input robustness defects were found and fixed: singleton tail batches now avoid BatchNorm failure, and numeric sample IDs are normalized before genotype, phenotype, relatedness, and split matching.
