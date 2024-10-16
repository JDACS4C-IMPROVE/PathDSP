import sys
import os
import polars as pl
import numpy as np
import pandas as pd
from functools import reduce
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from datetime import datetime
import RWR as rwr
import NetPEA as pea
import gseapy as gp
import sklearn.model_selection as skms
from sklearn.preprocessing import StandardScaler
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig #NCK
from improvelib.utils import str2bool #NCK
import improvelib.utils as frm #NCK
import improvelib.applications.drug_response_prediction.drug_utils as drugs #NCK
import improvelib.applications.drug_response_prediction.omics_utils as omics #NCK
import improvelib.applications.drug_response_prediction.drp_utils as drp #NCK

from model_params_def import pathdsp_preprocess_params

file_path = Path(__file__).resolve().parent

req_preprocess_args = [ll["name"] for ll in pathdsp_preprocess_params]

def mkdir(directory):
    directories = directory.split("/")
    folder = ""
    for d in directories:
        folder += d + "/"
        if not os.path.exists(folder):
            print("creating folder: %s" % folder)
            os.mkdir(folder)


def preprocess(params):
    for i in [
        "drug_bits_file",
        "dgnet_file",
        "mutnet_file",
        "cnvnet_file",
        "exp_file",
    ]:
        params[i] = params["output_dir"] + "/" + params[i]
    return params


# set timer
def cal_time(end, start):
    """return time spent"""
    # end = datetime.now(), start = datetime.now()
    datetimeFormat = "%Y-%m-%d %H:%M:%S.%f"
    spend = datetime.strptime(str(end), datetimeFormat) - datetime.strptime(
        str(start), datetimeFormat
    )
    return spend

def response_out(params, split_file):
    response_df = drp.DrugResponseLoader(params, split_file=split_file, verbose=True)
    return response_df.dfs["response.tsv"]


def smile2bits(params):
    start = datetime.now()
    response_df = [response_out(params, params[split_file]) for split_file in ["train_split_file", "test_split_file", "val_split_file"]]
    response_df = pd.concat(response_df, ignore_index=True)
    smile_df = drugs.DrugsLoader(params)
    smile_df = smile_df.dfs['drug_SMILES.tsv']
    smile_df = smile_df.reset_index()
    smile_df.columns = ["drug", "smile"]
    smile_df = smile_df.drop_duplicates(subset=["drug"], keep="first").set_index("drug")
    smile_df = smile_df.loc[smile_df.index.isin(response_df["improve_chem_id"]),]
    bit_int = params["bit_int"]
    record_list = []
    # smile2bits drug by drug
    n_drug = 1
    for idx, row in smile_df.iterrows():
        drug = idx
        smile = row["smile"]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        mbit = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=bit_int))
        # drug_mbit_dict.update({drug:mbit})
        # append to result
        record_list.append(tuple([drug] + mbit))
        if len(mbit) == bit_int:
            n_drug += 1
    print("total {:} drugs with bits".format(n_drug))
    # convert dict to dataframe
    colname_list = ["drug"] + ["mBit_" + str(i) for i in range(bit_int)]
    drug_mbit_df = pd.DataFrame.from_records(record_list, columns=colname_list)
    # drug_mbit_df = pd.DataFrame.from_dict(drug_mbit_dict, orient='index', columns=colname_list)
    # drug_mbit_df.index.name = 'drug'
    print("unique drugs={:}".format(len(drug_mbit_df["drug"].unique())))
    # save to file
    drug_mbit_df.to_csv(params["drug_bits_file"], header=True, index=False, sep="\t")
    print("[Finished in {:}]".format(cal_time(datetime.now(), start)))


def times_expression(rwr, exp):
    """
    :param rwrDf: dataframe of cell by gene probability matrix
    :param expDf: dataframe of cell by gene expression matrix
    :return rwr_timesexp_df: dataframe of cell by gene probability matrix,
                             in which genes are multiplied with expression values

    Note: this function assumes cells are all overlapped while gene maybe not
    """
    cell_list = sorted(list(set(rwr.index) & set(exp.index)))
    gene_list = sorted(list(set(rwr.columns) & set(exp.columns)))

    if len(cell_list) == 0:
        print("ERROR! no overlapping cell lines")
        sys.exit(1)
    if len(gene_list) == 0:
        print("ERROR! no overlapping genes")
        sys.exit(1)
    # multiply with gene expression for overlapping cell, gene
    rwr_timesexp = rwr.loc[cell_list, gene_list] * exp.loc[cell_list, gene_list]
    # concat with other gene
    out_gene_list = list(set(rwr.columns) - set(gene_list))
    out_df = pd.concat([rwr_timesexp, rwr[out_gene_list]], axis=1)
    return out_df


def run_netpea(params, dtype, multiply_expression):
    # timer
    start_time = datetime.now()
    ppi_path = params["input_supp_data_dir"] + "/STRING/9606.protein_name.links.v11.0.pkl"
    pathway_path = (
        params["input_supp_data_dir"] + "/MSigdb/union.c2.cp.pid.reactome.v7.2.symbols.gmt"
    )
    log_transform = False
    permutation_int = params["permutation_int"]
    seed_int = params["seed_int"]
    cpu_int = params["cpu_int"]
    response_df = [response_out(params, params[split_file]) for split_file in ["train_split_file", "test_split_file", "val_split_file"]]
    response_df = pd.concat(response_df, ignore_index=True)
    omics_data = omics.OmicsLoader(params)
    if dtype == "DGnet":
        drug_info = pd.read_csv(params["input_dir"] + "/x_data/drug_info.tsv", sep="\t")
        drug_info["NAME"] = drug_info["NAME"].str.upper()
        target_info = pd.read_csv(
            params["input_supp_data_dir"] + "/data/DB.Drug.Target.txt", sep="\t"
        )
        target_info = target_info.rename(columns={"drug": "NAME"})
        combined_df = pd.merge(drug_info, target_info, how="left", on="NAME").dropna(
            subset=["gene"]
        )
        combined_df = combined_df.loc[
            combined_df["improve_chem_id"].isin(response_df["improve_chem_id"]),
        ]
        restart_path = params["output_dir"] + "/drug_target.txt"
        combined_df.iloc[:, -2:].to_csv(
            restart_path, sep="\t", header=True, index=False
        )
        outpath = params["dgnet_file"]
    elif dtype == "MUTnet":
        mutation_data = omics_data.dfs['cancer_mutation_count.tsv']
        #mutation_data = mutation_data.reset_index()
        mutation_data = pd.melt(mutation_data, id_vars="improve_sample_id").loc[
            lambda x: x["value"] > 0
        ]
        mutation_data = mutation_data.loc[
            mutation_data["improve_sample_id"].isin(response_df["improve_sample_id"]),
        ]
        restart_path = params["output_dir"] + "/mutation_data.txt"
        mutation_data.iloc[:, 0:2].to_csv(
            restart_path, sep="\t", header=True, index=False
        )
        outpath = params["mutnet_file"]
    else:
        cnv_data = omics_data.dfs['cancer_discretized_copy_number.tsv']
        #cnv_data = cnv_data.reset_index()
        cnv_data = pd.melt(cnv_data, id_vars="improve_sample_id").loc[
            lambda x: x["value"] != 0
        ]
        cnv_data = cnv_data.loc[
            cnv_data["improve_sample_id"].isin(response_df["improve_sample_id"]),
        ]
        restart_path = params["output_dir"] + "/cnv_data.txt"
        cnv_data.iloc[:, 0:2].to_csv(restart_path, sep="\t", header=True, index=False)
        outpath = params["cnvnet_file"]
    # perform Random Walk
    print(datetime.now(), "performing random walk with restart")
    rwr_df = rwr.RWR(
        ppi_path,
        restart_path,
        restartProbFloat=0.5,
        convergenceFloat=0.00001,
        normalize="l1",
        weighted=True,
    ).get_prob()
    # multiply with gene expression
    if multiply_expression:
        print(
            datetime.now(),
            "multiplying gene expression with random walk probability for genes were expressed",
        )
        # exp_df = improve_utils.load_gene_expression_data(gene_system_identifier='Gene_Symbol')
        # exp_df = drp.load_omics_data(
        #     params,
        #     omics_type="gene_expression",
        #     canc_col_name="improve_sample_id",
        #     gene_system_identifier="Gene_Symbol",
        # )
        exp_df = omics_data.dfs['cancer_gene_expression.tsv']
        exp_df = exp_df.set_index(params['canc_col_name'])
        rwr_df = times_expression(rwr_df, exp_df)
    # rwr_df.to_csv(out_path+'.RWR.txt', header=True, index=True, sep='\t')
    # perform Pathwa Enrichment Analysis
    print(datetime.now(), "performing network-based pathway enrichment")
    cell_pathway_df = pea.NetPEA(
        rwr_df,
        pathway_path,
        log_transform=log_transform,
        permutation=permutation_int,
        seed=seed_int,
        n_cpu=cpu_int,
        out_path=outpath,
    )
    print("[Finished in {:}]".format(cal_time(datetime.now(), start_time)))


def prep_input(params):
    # Read data files
    drug_mbit_df = pd.read_csv(params["drug_bits_file"], sep="\t", index_col=0)
    drug_mbit_df = drug_mbit_df.reset_index().rename(columns={"drug": "drug_id"})
    DGnet = pd.read_csv(params["dgnet_file"], sep="\t", index_col=0)
    DGnet = (
        DGnet.add_suffix("_dgnet").reset_index().rename(columns={"index": "drug_id"})
    )
    CNVnet = pd.read_csv(params["cnvnet_file"], sep="\t", index_col=0)
    CNVnet = (
        CNVnet.add_suffix("_cnvnet")
        .reset_index()
        .rename(columns={"index": "sample_id"})
    )
    MUTnet = pd.read_csv(params["mutnet_file"], sep="\t", index_col=0)
    MUTnet = (
        MUTnet.add_suffix("_mutnet")
        .reset_index()
        .rename(columns={"index": "sample_id"})
    )
    EXP = pd.read_csv(params["exp_file"], sep="\t", index_col=0)
    EXP = EXP.add_suffix("_exp").reset_index().rename(columns={"index": "sample_id"})
    response_df = [response_out(params, params[split_file]) for split_file in ["train_split_file", "test_split_file", "val_split_file"]]
    response_df = pd.concat(response_df, ignore_index=True)
    response_df = response_df.rename(
        columns={"improve_chem_id": "drug_id", "improve_sample_id": "sample_id"}
    )
    # Extract relevant IDs
    common_drug_ids = reduce(
        np.intersect1d,
        (drug_mbit_df["drug_id"], DGnet["drug_id"], response_df["drug_id"]),
    )
    common_sample_ids = reduce(
        np.intersect1d,
        (
            CNVnet["sample_id"],
            MUTnet["sample_id"],
            EXP["sample_id"],
            response_df["sample_id"],
        ),
    )
    response_df = response_df.loc[
        (response_df["drug_id"].isin(common_drug_ids))
        & (response_df["sample_id"].isin(common_sample_ids)),
        :,
    ]
    drug_mbit_df = (
        drug_mbit_df.loc[drug_mbit_df["drug_id"].isin(common_drug_ids), :]
        .set_index("drug_id")
        .sort_index()
    )
    DGnet = (
        DGnet.loc[DGnet["drug_id"].isin(common_drug_ids), :]
        .set_index("drug_id")
        .sort_index()
    )
    CNVnet = (
        CNVnet.loc[CNVnet["sample_id"].isin(common_sample_ids), :]
        .set_index("sample_id")
        .sort_index()
    )
    MUTnet = (
        MUTnet.loc[MUTnet["sample_id"].isin(common_sample_ids), :]
        .set_index("sample_id")
        .sort_index()
    )
    EXP = (
        EXP.loc[EXP["sample_id"].isin(common_sample_ids), :]
        .set_index("sample_id")
        .sort_index()
    )

    drug_data = drug_mbit_df.join(DGnet)
    sample_data = CNVnet.join([MUTnet, EXP])
    ## export train,val,test set
    for i in ["train", "test", "val"]:
        response_df = drp.DrugResponseLoader(params, split_file=params[i+"_split_file"], verbose=True)
        response_df = response_df.dfs['response.tsv']
        response_df = response_df.rename(
            columns={"improve_chem_id": "drug_id", "improve_sample_id": "sample_id"}
        )
        response_df = response_df.loc[
            (response_df["drug_id"].isin(common_drug_ids))
            & (response_df["sample_id"].isin(common_sample_ids)),
            :,
        ]
        
        comb_data_mtx = pd.DataFrame(
            {
                "drug_id": response_df["drug_id"].values,
                "sample_id": response_df["sample_id"].values,
            }
        )
        comb_data_mtx = (
            comb_data_mtx.set_index(["drug_id", "sample_id"])
            .join(drug_data, on="drug_id")
            .join(sample_data, on="sample_id")
        )
        ss = StandardScaler()
        comb_data_mtx.iloc[:,params["bit_int"]:comb_data_mtx.shape[1]] = ss.fit_transform(comb_data_mtx.iloc[:,params["bit_int"]:comb_data_mtx.shape[1]])
        ## add 0.01 to avoid possible inf values
        comb_data_mtx["response"] = np.log10(response_df[params["y_col_name"]].values + 0.01)
        comb_data_mtx = comb_data_mtx.dropna()
        comb_data_mtx_to_save = pd.concat([comb_data_mtx["drug_id"], comb_data_mtx["sample_id"]], axis=1)
        auc_to_save = pd.Series(comb_data_mtx["response"])
        auc_to_save = auc_to_save.apply(lambda x: 10 ** (x) - 0.01)
        comb_data_mtx_to_save[params["y_col_name"]] = auc_to_save
        frm.save_stage_ydf(ydf=comb_data_mtx_to_save, stage=i, output_dir=params["output_dir"])
        pl.from_pandas(comb_data_mtx).write_csv(
            params["output_dir"] + "/" + frm.build_ml_data_file_name(data_format=params["data_format"], stage=i)
, separator="\t", has_header=True
        )


def run_ssgsea(params):
    # expMat = improve_utils.load_gene_expression_data(sep='\t')
    # expMat = drp.load_omics_data(
    #     params,
    #     omics_type="gene_expression",
    #     canc_col_name="improve_sample_id",
    #     gene_system_identifier="Gene_Symbol",
    # )
    omics_data = omics.OmicsLoader(params)
    expMat = omics_data.dfs['cancer_gene_expression.tsv']
    expMat = expMat.set_index(params['canc_col_name'])

    # response_df = improve_utils.load_single_drug_response_data(source=params['data_type'],
    #                                                     split=params['split'], split_type=["train", "test", "val"],
    #                                                     y_col_name=params['metric'])
    response_df = [response_out(params, params[split_file]) for split_file in ["train_split_file", "test_split_file", "val_split_file"]]
    response_df = pd.concat(response_df, ignore_index=True)
    expMat = expMat.loc[expMat.index.isin(response_df["improve_sample_id"]),]
    gct = expMat.T  # gene (rows) cell lines (columns)
    pathway_path = (
        params["input_supp_data_dir"] + "/MSigdb/union.c2.cp.pid.reactome.v7.2.symbols.gmt"
    )
    gmt = pathway_path
    tmp_str = params["output_dir"] + "/tmpdir_ssgsea/"

    if not os.path.isdir(tmp_str):
        os.mkdir(tmp_str)

    # run enrichment
    ssgsea = gp.ssgsea(
        data=gct,  # gct: a matrix of gene by sample
        gene_sets=gmt,  # gmt format
        outdir=tmp_str,
        scale=True,
        permutation_num=0,  # 1000
        no_plot=True,
        processes=params["cpu_int"],
        # min_size=0,
        format="png",
    )

    result_mat = ssgsea.res2d.T  # get the normalized enrichment score (i.e., NES)
    result_mat.to_csv(tmp_str + "ssGSEA.txt", header=True, index=True, sep="\t")

    f = open(tmp_str + "ssGSEA.txt", "r")
    lines = f.readlines()
    total_dict = {}
    for cell in set(lines[1].split()):
        total_dict[cell] = {}
    cell_lines = lines[1].split()
    vals = lines[4].split()
    for i, pathway in enumerate((lines[2].split())):
        if i > 0:
            total_dict[cell_lines[i]][pathway] = float(vals[i])
    df = pd.DataFrame(total_dict)
    df.T.to_csv(params["exp_file"], header=True, index=True, sep="\t")

def run(params):
    frm.create_outdir(outdir=params["output_dir"])
    params = preprocess(params)
    print("convert drug to bits.")
    smile2bits(params)
    print("compute DGnet.")
    run_netpea(params, dtype="DGnet", multiply_expression=False)
    print("compute MUTnet.")
    run_netpea(params, dtype="MUTnet", multiply_expression=True)
    print("compute CNVnet.")
    run_netpea(params, dtype="CNVnet", multiply_expression=True)
    print("compute EXP.")
    run_ssgsea(params)
    print("prepare final input file.")
    prep_input(params)


def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        file_path, 
        default_config="PathDSP_params.txt", 
        additional_definitions=pathdsp_preprocess_params)
    run(params)


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1:])
    print("[Preprocessing finished in {:}]".format(cal_time(datetime.now(), start)))
