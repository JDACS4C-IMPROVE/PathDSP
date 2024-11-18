"""
Before running this script, first need to preprocess the data.
This can be done by running preprocess_example.sh

It is assumed that the csa benchmark data is downloaded via download_csa.sh
and the env vars $PYTHONPATH is set:
export PYTHONPATH=$PYTHONPATH:/path/to/IMPROVE_lib

It also assumes that your processed training data is at: "ml_data/{source}-{source}/split_{split}"
validation data is at: "ml_data/{source}-{source}/split_{split}"
model output files will be saved at "dh_hpo_improve/{source}/split_{split}"

mpirun -np 10 python hpo_subprocess.py
"""
# import copy
import json
import subprocess
import pandas as pd
import os
import time
from pathlib import Path
import logging
import mpi4py
from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from mpi4py import MPI
import socket
import hpo_deephyper_params_def
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig



# ---------------------
# Enable using multiple GPUs
# ---------------------

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#local_rank = os.environ["PMI_LOCAL_RANK"]

# CUDA_VISIBLE_DEVICES is now set via set_affinity_gpu_polaris.sh
# uncomment the below commands if running via interactive node
num_gpus_per_node = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_gpus_per_node)
cuda_name = "cuda:" + str(rank % num_gpus_per_node)

# ---------------------
# Enable logging
# ---------------------

logging.basicConfig(
    # filename=f"deephyper.{rank}.log, # optional if we want to store the logs to disk
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

# ---------------------
# Hyperparameters
# ---------------------
problem = HpProblem()

problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=64)
problem.add_hyperparameter((1e-6, 1e-2, "log-uniform"),
                           "learning_rate", default_value=0.001)
# problem.add_hyperparameter((0, 0.5), "dropout", default_value=0.0)
# problem.add_hyperparameter([True, False], "early_stopping", default_value=False)


@profile
def run(job, optuna_trial=None):
    model_outdir_job_id = Path(params['model_outdir'] + f"/{job.id}")
    learning_rate = job.parameters["learning_rate"]
    batch_size = job.parameters["batch_size"]

    print(f"Launching run: batch_size={batch_size}, learning_rate={learning_rate}")
    subprocess_res = subprocess.run(
        [
            "bash", 
            "hpo_deephyper_subprocess_train.sh",
             str(params['model_environment']),
             str(params['script_name']),
             str(params['ml_data_dir']),
             str(model_outdir_job_id),
             str(learning_rate),
             str(batch_size),
             str(params['epochs']),
             str(os.environ["CUDA_VISIBLE_DEVICES"])
        ], 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    # Logger
    print(f"returncode = {subprocess_res.returncode}")
    result_file_name_stdout = model_outdir_job_id / 'logs.txt'
    if model_outdir_job_id.exists() is False: # If subprocess fails, model_dir may not be created and we need to write the log files in model_dir
        os.makedirs(model_outdir_job_id, exist_ok=True)
    with open(result_file_name_stdout, 'w') as file:
        file.write(subprocess_res.stdout)

    # Load val_scores and get val_loss
    f = open(model_outdir_job_id / "val_scores.json")
    val_scores = json.load(f)
    objective = -val_scores[params['val_loss']]

    # Checkpoint the model weights
    with open(f"{params['output_dir']}/model_{job.id}.pkl", "w") as f:
        f.write("model weights")

    # return score
    return {"objective": objective, "metadata": val_scores}


if __name__ == "__main__":
    # Initialize parameters for DeepHyper HPO
    filepath = Path(__file__).resolve().parent
    cfg = DRPPreprocessConfig() 
    global params
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="hpo_deephyper_params.ini",
        additional_definitions=hpo_deephyper_params_def.additional_definitions
    )
    output_dir = Path(params['output_dir'])
    if output_dir.exists() is False:
        os.makedirs(output_dir, exist_ok=True)
    params['ml_data_dir'] = f"ml_data/{params['source']}-{params['source']}/split_{params['split']}"
    params['model_outdir'] = f"{params['output_dir']}/{params['source']}/split_{params['split']}"
    params['script_name'] = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
    
    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:

        if evaluator is not None:
            print(problem)
            search = CBO(
                problem,
                evaluator,
                log_dir=params['output_dir'],
                verbose=1,
            )
            results = search.search(max_evals=params['max_evals'])
            results = results.sort_values(f"m:{params['val_loss']}", ascending=True)
            results.to_csv(f"{params['output_dir']}/hpo_results.csv", index=False)
    print("current node: ", socket.gethostname(), "; current rank: ", rank, "; CUDA_VISIBLE_DEVICE is set to: ", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Finished deephyper HPO.")
    print(params['max_evals'])
