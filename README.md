# Run HPO using deephyper on Polaris

## Install conda environment for the curated model (PathDSP)

```
## install PathDSP
git clone -b deephyper https://github.com/Liuy12/PathDSP.git
## install IMPROVE
git clone -b develop https://github.com/JDACS4C-IMPROVE/IMPROVE.git
## define where to install PathDSP env
export PathDSP_env=./PathDSP_env/
conda env create -f ./PathDSP/environment_082223.yml -p $PathDSP_env
conda activate ${PathDSP_env}
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

## Download csa benchmark data

```
wget --cut-dirs=7 -P ./ -nH -np -m ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data
```

## Download additional author data (PathDSP only)

```
mkdir author_data
bash ./PathDSP/download_author_data.sh author_data/
```

## Define environment variables

```
### if necessary, request an interactive node from polaris to testing purposes
###  qsub -A IMPROVE -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug
### NEED to cd into your working directory again once the job started
improve_lib="$PWD/IMPROVE/"
pathdsp_lib="$PWD/PathDSP/"
# notice the extra PathDSP folder after pathdsp_lib
echo "export PYTHONPATH=$PYTHONPATH:${improve_lib}:${pathdsp_lib}/PathDSP/" >> IMPROVE_env
# IMPROVE_DATA_DIR
echo "export IMPROVE_DATA_DIR=$PWD/csa_data/" >> IMPROVE_env
# AUTHOR_DATA_DIR required for PathDSP
echo "export AUTHOR_DATA_DIR=$PWD/author_data/" >> IMPROVE_env
# PathDSP_env: conda environment path for the model
echo "export PathDSP_env=$PathDSP_env" >> IMPROVE_env
source $PWD/IMPROVE_env
```

## Perform preprocessing

```
conda activate $PathDSP_env
## You can copy the processed files for PathDSP
cp -r /lus/eagle/projects/IMPROVE_Aim1/yuanhangl_alcf/PathDSP/ml_data/ ./PathDSP/
## Alternatively, run the preprocess script
## This script taks around 40 mins to complete
## python PathDSP/PathDSP_preprocess_improve.py --ml_data_outdir=./PathDSP/ml_data/GDSCv1-GDSCv1/split_4/
```

## Perform HPO across two nodes based on conda

```
cd PathDSP
# supply environment variables to qsub
qsub -v IMPROVE_env=../IMPROVE_env ./hpo_scale.sh
## for interactive node, you can run: mpirun -np 10 python hpo_subprocess.py
```

## Alternatively, perform HPO using singularity container across two nodes

```
## copy processed to IMPROVE_DATA_DIR
cp -r /lus/eagle/projects/IMPROVE_Aim1/yuanhangl_alcf/PathDSP/ml_data/ $IMPROVE_DATA_DIR
## specify singularity image file for PathDSP
echo "export PathDSP_sif=/lus/eagle/projects/IMPROVE_Aim1/yuanhangl_alcf/PathDSP.sif" >> IMPROVE_env
cd PathDSP
qsub -v IMPROVE_env=../IMPROVE_env ./hpo_scale_singularity.sh
## for interative node, run: mpirun -np 10 python hpo_subprocess_singularity.py
```