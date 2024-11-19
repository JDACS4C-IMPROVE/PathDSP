# Run HPO using DeepHyper on Lambda with conda

## Install conda environment for the curated model (PathDSP)
Install PathDSP:
```
cd <working_dir>
git clone https://github.com/JDACS4C-IMPROVE/PathDSP
```

Install IMPROVE and download data:
```
cd PathDSP
source setup_improve.sh
```

Install PathDSP environment:
```
cd PathDSP
export PathDSP_env=./PathDSP_env/
conda env create -f PathDSP_env_conda.yml -p $PathDSP_env
```

## Perform preprocessing
Run the preprocess script. This script taks around 40 mins to complete.
The workflow assumes that your preprocessed data is at: "ml_data/{source}-{source}/split_{split}"

```
cd PathDSP
conda activate $PathDSP_env
python PathDSP_preprocess_improve.py --input_dir ./csa_data/raw_data --output_dir ./ml_data/CCLE-CCLE/split_0
conda deactivate
```

## Install conda environment for DeepHyper
```
module load openmpi
conda create -n dh python=3.9 -y
conda activate dh
conda install gxx_linux-64 gcc_linux-64
pip install "deephyper[default]"
pip install mpi4py
```

## Perform HPO
If necesssary, activate environment:
```
module load openmpi 
conda activate dh
export PYTHONPATH=../IMPROVE
```

Run HPO:
```
mpirun -np 10 python hpo_deephyper_subprocess.py
```

# Run HPO using DeepHyper on Polaris with conda

# Run HPO using DeepHyper on Polaris with singularity

