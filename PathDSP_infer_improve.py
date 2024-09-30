import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import torch as tch
import torch.utils.data as tchud
import polars as pl
import model_utils.myModel as mynet
import model_utils.myDataloader as mydl
import model_utils.myUtility as myutil

from PathDSP_preprocess_improve import mkdir, preprocess
from PathDSP_train_improve import (
    predicting,
    cal_time,
)
from improvelib.applications.drug_response_prediction.config import DRPInferConfig #NCK
import improvelib.utils as frm #NCK
from model_params_def import pathdsp_infer_params

file_path = os.path.dirname(os.path.realpath(__file__))


def run(params):   
    frm.create_outdir(outdir=params["output_dir"])
    #params =  preprocess(params)
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    test_df = pl.read_csv(params["input_data_dir"] + "/" + test_data_fname, separator = "\t").to_pandas()
    Xtest_arr = test_df.iloc[:, 0:-1].values
    ytest_arr = test_df.iloc[:, -1].values
    Xtest_arr = np.array(Xtest_arr).astype('float32')
    ytest_arr = np.array(ytest_arr).astype('float32')
    trained_net = mynet.FNN(Xtest_arr.shape[1])
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"], model_file_format=params["model_file_format"], model_dir=params["input_model_dir"])
    trained_net.load_state_dict(tch.load(modelpath))
    trained_net.eval()
    #myutil.set_seed(params["seed_int"])
    cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_env_visible is not None:
        device = 'cuda:0'
    else:
        device = myutil.get_device(uth=int(params['cuda_name'].split(':')[1]))
    test_dataset = mydl.NumpyDataset(tch.from_numpy(Xtest_arr), tch.from_numpy(ytest_arr))
    test_dl = tchud.DataLoader(test_dataset, batch_size=params['infer_batch'], shuffle=False)
    start = datetime.now()
    test_true, test_pred = predicting(trained_net, device, data_loader=test_dl)

    test_true_untrans = test_true.apply(lambda x: 10 ** (x) - 0.01)
    test_pred_untrans = test_pred.apply(lambda x: 10 ** (x) - 0.01)

    frm.store_predictions_df(
        y_true=test_true_untrans,
        y_pred=test_pred_untrans, 
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true_untrans, 
            y_pred=test_pred_untrans, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    print('Inference time :[Finished in {:}]'.format(cal_time(datetime.now(), start)))
    return True

def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        file_path, 
        default_config="PathDSP_params.txt", 
        additional_definitions=pathdsp_infer_params)
    if_ran = run(params)
    print("\nFinished inference of PathDSP model.")

    
if __name__ == "__main__":
    main(sys.argv[1:])
