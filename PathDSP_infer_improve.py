import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import torch as tch
import torch.utils.data as tchud
import polars as pl
import myModel as mynet
import myDataloader as mydl
import myUtility as myutil

from PathDSP_preprocess_improve import mkdir, preprocess
from PathDSP_train_improve import (
    predicting,
    preprocess,
    cal_time,
    metrics_list,
)
from improvelib.applications.drug_response_prediction.config import DRPInferConfig #NCK
import improvelib.utils as frm #NCK

file_path = os.path.dirname(os.path.realpath(__file__))


def run(params):
    if "input_data_dir" in params:
        data_dir = params["input_data_dir"]
    else:
        data_dir = params["input_dir"]
    if "input_model_dir" in params:
        model_dir = params["input_model_dir"]
    else:
        model_dir = params["input_dir"]    
    frm.create_outdir(outdir=params["output_dir"])
    params =  preprocess(params)
    test_data_fname = frm.build_ml_data_name(params, stage="test")
    test_df = pl.read_csv(data_dir + "/" + test_data_fname, separator = "\t").to_pandas()
    Xtest_arr = test_df.iloc[:, 0:-1].values
    ytest_arr = test_df.iloc[:, -1].values
    Xtest_arr = np.array(Xtest_arr).astype('float32')
    ytest_arr = np.array(ytest_arr).astype('float32')
    trained_net = mynet.FNN(Xtest_arr.shape[1])
    modelpath = frm.build_model_path(params, model_dir=model_dir)
    trained_net.load_state_dict(tch.load(modelpath))
    trained_net.eval()
    myutil.set_seed(params["seed_int"])
    cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_env_visible is not None:
        device = 'cuda:0'
    else:
        device = myutil.get_device(uth=int(params['cuda_name'].split(':')[1]))
    test_dataset = mydl.NumpyDataset(tch.from_numpy(Xtest_arr), tch.from_numpy(ytest_arr))
    test_dl = tchud.DataLoader(test_dataset, batch_size=params['test_batch'], shuffle=False)
    start = datetime.now()
    test_true, test_pred = predicting(trained_net, device, data_loader=test_dl)
    frm.store_predictions_df(
        params, y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["output_dir"]
    )
    test_scores = frm.compute_performace_scores(
        params, y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["output_dir"], metrics=metrics_list
    )
    print('Inference time :[Finished in {:}]'.format(cal_time(datetime.now(), start)))
    return test_scores

def main(args):
    cfg = DRPInferConfig() #NCK
    params = cfg.initialize_parameters(file_path, default_config="PathDSP_default_model.txt", additional_definitions=None, required=None) #NCK
    test_scores = run(params)
    print("\nFinished inference of PathDSP model.")

    
if __name__ == "__main__":
    main(sys.argv[1:])
