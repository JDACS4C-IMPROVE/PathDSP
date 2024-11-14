# Run HPO using deephyper on Polaris

## Install conda environment for the curated model (PathDSP)
```
## install PathDSP
git clone https://github.com/JDACS4C-IMPROVE/PathDSP
cd PathDSP
git checkout develop

## install IMPROVE and download data
source setup_improve.sh
or 
export PYTHONPATH=../IMPROVE

## define where to install PathDSP env
export PathDSP_env=./PathDSP_env/
conda env create -f PathDSP_env_conda.yml -p $PathDSP_env

## set up environment variables
cd ..
cd <working_dir>
improve_lib="$PWD/IMPROVE/"
echo "export PYTHONPATH=$PYTHONPATH:${improve_lib}" >> IMPROVE_env
echo "export PathDSP_env=$PathDSP_env" >> IMPROVE_env
source $PWD/IMPROVE_env
```



## Perform preprocessing
Run the preprocess script. This script taks around 40 mins to complete.
The workflow assumes that your preprocessed data is at: "ml_data/{source}-{source}/split_{split}"

```
### if necessary, request an interactive node from polaris to testing purposes
###  qsub -A IMPROVE -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug
### NEED to cd into your working directory again once the job started
```

```
cd PathDSP
conda activate $PathDSP_env
python PathDSP_preprocess_improve.py --input_dir ./csa_data/raw_data --output_dir ./ml_data/CCLE-CCLE/split_0
```

## Perform HPO using singularity container across two nodes
This will presumably have to be redone for alpha.

```
## copy processed to IMPROVE_DATA_DIR
cp -r /lus/eagle/projects/IMPROVE_Aim1/yuanhangl_alcf/PathDSP/ml_data/ $IMPROVE_DATA_DIR
## specify singularity image file for PathDSP
echo "export PathDSP_sif=/lus/eagle/projects/IMPROVE_Aim1/yuanhangl_alcf/PathDSP.sif" >> IMPROVE_env
cd PathDSP
## submit to debug queue
qsub -v IMPROVE_env=../IMPROVE_env ./hpo_scale_singularity_debug.sh
## to submit to debug-scaling or prod queue
## use hpo_scale_singularity_debug_scaling.sh 
## or hpo_scale_singularity_prod.sh
## for interative node, run: mpirun -np 10 python hpo_subprocess_singularity.py
```

## Alternatively, perform HPO across two nodes based on conda

```
cd PathDSP
# supply environment variables to qsub
qsub -v IMPROVE_env=../IMPROVE_env ./hpo_scale.sh
## for interactive node, you can run: mpirun -np 10 python hpo_deephyper_subprocess.py
```