# PathDSP

This repository demonstrates how to use the [IMPROVE library v0.0.3-beta](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.0.3-beta) for building a drug response prediction (DRP) model using PathDSP, and provides examples with the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

This version, tagged as `v0.0.3-beta`, is the final release before transitioning to `v0.1.0-alpha`, which introduces a new API. Version `v0.0.3-beta` and all previous releases have served as the foundation for developing essential components of the IMPROVE software stack. Subsequent releases build on this legacy with an updated API, designed to encourage broader adoption of IMPROVE and its curated models by the research community.

A more detailed tutorial can be found [here](https://jdacs4c-improve.github.io/docs/v0.0.3-beta/content/ModelContributorGuide.html).


## Dependencies
Installation instuctions are detailed below in [Step-by-step instructions](#step-by-step-instructions).

Conda `yml` file [environment_082223.yml](./environment_082223.yml)

ML framework:
+ [Torch](https://pytorch.org/) -- deep learning framework for building the prediction model

IMPROVE dependencies:
+ [IMPROVE v0.0.3-beta](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.0.3-beta)
+ [candle_lib](https://github.com/ECP-CANDLE/candle_lib) - IMPROVE dependency (enables various hyperparameter optimization on HPC machines) 



## Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```


## Model scripts and parameter file
+ `PathDSP_preprocess_improve.py` - takes benchmark data files and transforms into files for training and inference
+ `PathDSP_train_improve.py` - trains the PathDSP model
+ `PathDSP_infer_improve.py` - runs inference with the trained PathDSP model
+ `PathDSP_default_model.txt` - default parameter file



# Step-by-step instructions

### 1. Clone the model repository
```
git clone https://github.com/JDACS4C-IMPROVE/PathDSP
cd PathDSP
git checkout v0.0.3-beta
```


### 2. Set computational environment
Create conda env using `yml`
```
conda env create -f environment_082223.yml -n PathDSP_env
conda activate PathDSP_env
```


### 3. Run `setup_improve.sh`.
```bash
source setup_improve.sh
```

This will:
1. Download cross-study analysis (CSA) benchmark data into `./csa_data/`.
2. Clone IMPROVE repo (checkout tag `v0.0.3-beta`) outside the PathDSP model repo
3. Set up env variables: `IMPROVE_DATA_DIR` (to `./csa_data/`) and `PYTHONPATH` (adds IMPROVE repo).
4. Download the model-specific supplemental data (aka author data) and set up the env variable `AUTHOR_DATA_DIR`.


### 4. Preprocess CSA benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python PathDSP_preprocess_improve.py
```

Preprocesses the CSA data and creates train, validation (val), and test datasets.

Generates:
* three model input data files: `train_data.txt`, `val_data.txt`, `test_data.txt`

```
ml_data
└── gCSI
    └── split_0
        ├── tmpdir_ssgsea
        ├── EXP.txt
        ├── cnv_data.txt
        ├── CNVnet.txt
        ├── DGnet.txt
        ├── MUTnet.txt
        ├── drug_mbit_df.txt
        ├── drug_target.txt
        ├── mutation_data.txt 
        ├── test_data.txt
        ├── train_data.txt
        ├── val_data.txt
        └── x_data_gene_expression_scaler.gz
```


### 5. Train PathDSP model
```bash
python PathDSP_train_improve.py
```

Trains PathDSP using the model input data: `train_data.txt` (training), `val_data.txt` (for early stopping).

Generates:
* trained model: `model.pt`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
out_models
└── gCSI
    └── split_0
        ├── model.pt
        ├── checkpoint.pt
        ├── Val_Loss_orig.txt
        ├── val_scores.json
        └── val_y_data_predicted.csv
```


### 6. Run inference on test data with the trained model
```python PathDSP_infer_improve.py```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
out_infer
└── gCSI-gCSI
    └── split_0
        ├── test_scores.json
        └── test_y_data_predicted.csv
```