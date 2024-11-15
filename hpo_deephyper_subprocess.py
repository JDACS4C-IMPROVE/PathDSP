"""
Before running this script, first need to preprocess the data.
This can be done by running preprocess_example.sh

It is assumed that the csa benchmark data is downloaded via download_csa.sh
and the env vars $IMPROVE_DATA_DIR and $PYTHONPATH are set:
export IMPROVE_DATA_DIR="./csa_data/"
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
print(rank)
size = comm.Get_size()
print(size)
#NCK local_rank = os.environ["PMI_LOCAL_RANK"]

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
def prepare_parameters():
    # Initialize parameters for DeepHyper HPO
    filepath = Path(__file__).resolve().parent
    cfg = DRPPreprocessConfig() 
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="hpo_deephyper_params.ini",
        additional_definitions=hpo_deephyper_params_def.additional_definitions
    )

    params['ml_data_dir'] = f"ml_data/{params['source']}-{params['source']}/split_{params['split']}"
    params['model_outdir'] = f"{params['output_dir']}/{params['source']}/split_{params['split']}"
    params['log_dir'] = f"{params['output_dir']}_logs/"
    # subprocess_bashscript = "subprocess_train.sh"
    params['script_name'] = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
    print("NATASHA LOOK HERE")
    print(params)
    print("NATASHA DONE LOOK HERE")
    return params

@profile
def run(job, optuna_trial=None):

    # config = copy.deepcopy(job.parameters)
    # params = {
    #     "epochs": DEEPHYPER_BENCHMARK_MAX_EPOCHS,
    #     "timeout": DEEPHYPER_BENCHMARK_TIMEOUT,
    #     "verbose": False,
    # }
    # if len(config) > 0:
    #     remap_hyperparameters(config)
    #     params.update(config)

    model_outdir_job_id = params['model_outdir'] + f"/{job.id}"
    learning_rate = job.parameters["learning_rate"]
    batch_size = job.parameters["batch_size"]
    # val_scores = main_train_grapdrp([
    #     "--train_ml_data_dir", str(train_ml_data_dir),
    #     "--val_ml_data_dir", str(val_ml_data_dir),
    #     "--model_outdir", str(model_outdir_job_id),
    # ])
    print("model env:", params['model_environment'])
    print("script_name:", params['script_name'])
    print("ml_data_dir:", params['ml_data_dir'])
    print("model_outdir_job_id:", model_outdir_job_id)
    print("learning_rate:", learning_rate)
    print("batch_size:", batch_size)
    print("params['epochs']:", params['epochs'])
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("launch run")
    subprocess_res = subprocess.run(
        [
            "bash", 
            "subprocess_train.sh",
             str(params['model_environment']),
             str(params['script_name']),
             str(params['ml_data_dir']),
             str(model_outdir_job_id),
             str(learning_rate),
             str(batch_size),
             str(params['epochs']),
             #str(cuda_name)
             str(os.environ["CUDA_VISIBLE_DEVICES"])
        ], 
        capture_output=True, text=True, check=True
    )
    # Logger
    print(f"returncode = {subprocess_res.returncode}")
    result_file_name_stdout = model_outdir_job_id / 'logs.txt'
    if model_outdir_job_id.exists() is False: # If subprocess fails, model_dir may not be created and we need to write the log files in model_dir
        os.makedirs(model_outdir_job_id, exist_ok=True)
    with open(result_file_name_stdout, 'w') as file:
        file.write(subprocess_res.stdout)
    # print(subprocess_res.stdout)
    # print(subprocess_res.stderr)

    # Load val_scores and get val_loss
    # f = open(model_outdir + "/val_scores.json")
    f = open(model_outdir_job_id + "/val_scores.json")
    val_scores = json.load(f)
    objective = -val_scores["val_loss"]
    # print("objective:", objective)

    # Checkpoint the model weights
    with open(f"{log_dir}/model_{job.id}.pkl", "w") as f:
        f.write("model weights")

    # return score
    return {"objective": objective, "metadata": val_scores}


if __name__ == "__main__":
    # Start time
    start_full_wf = time.time()
    global params
    params = prepare_parameters()
    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:

        if evaluator is not None:
            print(problem)

            search = CBO(
                problem,
                evaluator,
                log_dir=params['log_dir'],
                verbose=1,
            )

            # max_evals = 2
            # max_evals = 4
            # max_evals = 10
            # max_evals = 20
            max_evals = 10
            # max_evals = 100
            results = search.search(max_evals=max_evals)
            results = results.sort_values("m:val_loss", ascending=True)
            results.to_csv(model_outdir + "/hpo_results.csv", index=False)
    #print("current node: ", socket.gethostname(), "; current rank: ", rank, "; local rank", local_rank, "; CUDA_VISIBLE_DEVICE is set to: ", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Finished deephyper HPO.")
