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
conda env create -f environment_082223.yml -p $PathDSP_env
conda activate $PathDSP_env
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

Download additional author data (PathDSP only)

```
mkdir author_data
bash ./PathDSP/download_author_data.sh author_data/
```

Define environment variables

```
### need to request an interactive node first from polaris
### use debug queue for testing
improve_lib="/path/to/IMPROVE/repo/"
pathdsp_lib="/path/to/pathdsp/repo/"
# notice the extra PathDSP folder after pathdsp_lib
export PYTHONPATH=$PYTHONPATH:${improve_lib}:${pathdsp_lib}/PathDSP/
export IMPROVE_DATA_DIR="/path/to/csa_data/"
export AUTHOR_DATA_DIR="/path/to/author_data/"
```

Activate deephyper environment and perform HPO

```
bash ./activate-dhenv.sh
cd PathDSP
mpirun -np 10 python hpo_subprocess.py
```