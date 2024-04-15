# Setup environment on Polaris for deephyper

Install conda environment for deephyper

```
git clone -b deephyper https://github.com/Liuy12/PathDSP.git
bash ./PathDSP/install_polaris.sh
```

Install conda environment for the curated model (PathDSP)

```
## install IMPROVE
git clone -b develop https://github.com/JDACS4C-IMPROVE/IMPROVE.git
## define where to install PathDSP env
export PathDSP_env=./PathDSP_env/
conda env create -f ./PathDSP/environment_082223.yml -p $PathDSP_env
conda activate ${PathDSP_env}
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

Download csa benchmark data

```
wget --cut-dirs=7 -P ./ -nH -np -m ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data
```

Download additional author data (PathDSP only)

```
mkdir author_data
bash ./PathDSP/download_author_data.sh author_data/
```

Define environment variables

```
### need to firstly request an interactive node first from polaris
### use debug queue for testing
### it might take a while for a node to become available
qsub -A IMPROVE -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug
### NEED to cd into your working directory again once the job started
improve_lib="$PWD/IMPROVE/"
pathdsp_lib="$PWD/PathDSP/"
# notice the extra PathDSP folder after pathdsp_lib
export PYTHONPATH=$PYTHONPATH:${improve_lib}:${pathdsp_lib}/PathDSP/
export IMPROVE_DATA_DIR="$PWD/csa_data/"
export AUTHOR_DATA_DIR="$PWD/author_data/"
export PathDSP_env="$PWD/PathDSP_env/"
```

Perform preprocessing

```
conda activate $PathDSP_env
## You can copy the processed files under my home dir
cp -r /lus/eagle/projects/IMPROVE_Aim1/yuanhangl_alcf/PathDSP/ml_data/ ./PathDSP/
## Alternatively, run the preprocess script
## This script taks around 40 mins to complete
## python PathDSP/PathDSP_preprocess_improve.py --ml_data_outdir=./PathDSP/ml_data/GDSCv1-GDSCv1/split_4/
```

Activate deephyper environment and perform HPO

```
# the .sh script sometimes does not activate the environment somehow
# bash ./activate-dhenv.sh
module load PrgEnv-gnu/8.3.3
module load conda/2023-10-04
conda activate ./dhenv/
cd PathDSP
mpirun -np 10 python hpo_subprocess.py
```
