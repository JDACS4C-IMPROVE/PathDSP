import sys
import os
import polars as pl
import numpy as np
import pandas as pd
import copy
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


def check_smiles_RDKit(smile_df, col_name):
    bad_smiles = {}
    good_smiles = {}
    for idx, row in smile_df.iterrows():
        mol = Chem.MolFromSmiles(row["smile"])
        if mol is None:
            bad_smiles[idx] = row['smile']
        else:
            good_smiles[idx] = row['smile']
    bad_smiles = pd.DataFrame.from_dict(bad_smiles, orient='index', columns=['smile'])
    good_smiles = pd.DataFrame.from_dict(good_smiles, orient='index', columns=['smile'])
    bad_smiles.index.name = col_name
    good_smiles.index.name = col_name
    print("bad smiles:", bad_smiles)
    print("good smiles:", good_smiles)
    return bad_smiles, good_smiles


def smile2bits(params, smile_df):
    record_list = []
    # smile2bits drug by drug
    n_drug = 1
    for idx, row in smile_df.iterrows():
        mol = Chem.MolFromSmiles(row["smile"])
        mbit = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=params['bit_int']))
        record_list.append(tuple([idx] + mbit))
    # convert dict to dataframe
    colname_list = [params['drug_col_name']] + ["mBit_" + str(i) for i in range(params['bit_int'])]
    drug_mbit_df = pd.DataFrame.from_records(record_list, columns=colname_list)
    # save to file
    #drug_mbit_df.to_csv(params["drug_bits_file"], header=True, index=False, sep="\t")
    return drug_mbit_df



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



def run_ssgsea(params, expMat, response_df):
    expMat = expMat.loc[expMat.index.isin(response_df[params['canc_col_name']]),]
    gct = expMat.T  # gene (rows) cell lines (columns)
    pathway_path = (params["input_supp_data_dir"] + "/MSigdb/union.c2.cp.pid.reactome.v7.2.symbols.gmt")
    #tmp_str = params["output_dir"] + "/tmpdir_ssgsea/"
    if not os.path.isdir(params["output_dir"] + "/tmpdir_ssgsea/"):
        os.mkdir(params["output_dir"] + "/tmpdir_ssgsea/")

    # run enrichment
    ssgsea = gp.ssgsea(data=gct,  # gct: a matrix of gene by sample
                       gene_sets=pathway_path,  # gmt format
                       outdir=params["output_dir"] + "/tmpdir_ssgsea/",
                       scale=True,
                       permutation_num=0,  # 1000
                       no_plot=True,
                       processes=params["cpu_int"],
                       format="png")

    result_mat = ssgsea.res2d.T  # get the normalized enrichment score (i.e., NES)
    result_mat.to_csv(params["output_dir"] + "/tmpdir_ssgsea/" + "ssGSEA.txt", header=True, index=True, sep="\t")

    f = open(params["output_dir"] + "/tmpdir_ssgsea/" + "ssGSEA.txt", "r")
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
    #df.T.to_csv(params["exp_file"], header=True, index=True, sep="\t")
    return df.T




def run(params):
    ppi_path = params["input_supp_data_dir"] + "/STRING/9606.protein_name.links.v11.0.pkl"
    pathway_path = (params["input_supp_data_dir"] + "/MSigdb/union.c2.cp.pid.reactome.v7.2.symbols.gmt")
    for i in ["drug_bits_file", "dgnet_file", "mutnet_file", "cnvnet_file", "exp_file",]:
        params[i] = params["output_dir"] + "/" + params[i]
    response_dfs = []
    for split_file in ["train_split_file", "test_split_file", "val_split_file"]:
        resp = drp.DrugResponseLoader(params, split_file=params[split_file], verbose=True)
        response_dfs = response_dfs + [resp.dfs["response.tsv"]]
    response_df = pd.concat(response_dfs, ignore_index=True)
    smile_df = drugs.DrugsLoader(params)
    smile_df = smile_df.dfs['drug_SMILES.tsv']
    smile_df = smile_df.reset_index()
    smile_df.columns = [params['drug_col_name'], "smile"]
    smile_df = smile_df.drop_duplicates(subset=[params['drug_col_name']], keep="first").set_index(params['drug_col_name'])

    smile_df = smile_df.loc[smile_df.index.isin(response_df["improve_chem_id"]),]
    omics_data = omics.OmicsLoader(params)
    mutation_data = omics_data.dfs['cancer_mutation_count.tsv']
    cnv_data = omics_data.dfs['cancer_discretized_copy_number.tsv']
    exp_df = omics_data.dfs['cancer_gene_expression.tsv']
    exp_df = exp_df.set_index(params['canc_col_name'])

    print("Convert drug to bits...")
    bad_smiles, good_smiles = check_smiles_RDKit(smile_df, params['drug_col_name'])
    drug_mbit_df = smile2bits(params, good_smiles)
    print("...finished drug to bits.")

    print("Compute DGnet...")
    drug_info = pd.read_csv(params["input_dir"] + "/x_data/drug_info.tsv", sep="\t")
    drug_info["NAME"] = drug_info["NAME"].str.upper()
    target_info = pd.read_csv(params["input_supp_data_dir"] + "/data/DB.Drug.Target.txt", sep="\t")
    target_info = target_info.rename(columns={"drug": "NAME"})
    combined_df = pd.merge(drug_info, target_info, how="left", on="NAME").dropna(subset=["gene"])
    combined_df = combined_df.loc[combined_df[params['drug_col_name']].isin(response_df[params['drug_col_name']]),]
    combined_df.iloc[:, -2:].to_csv(params["output_dir"] + "/drug_target.txt", sep="\t", header=True, index=False)
    # Perform random walk with restart
    DGnet_rwr_df = rwr.RWR(
        ppiPathStr=ppi_path,
        restartPathStr=params["output_dir"] + "/drug_target.txt",
        restartProbFloat=0.5,
        convergenceFloat=0.00001,
        normalize="l1",
        weighted=True).get_prob()
    DGnet_rwr_df.to_csv("DGnet_rwr_df.tsv", sep='\t')
    DGnet = pea.NetPEA(
        rwrPath=DGnet_rwr_df,
        pathwayGMT=pathway_path,
        log_transform=False,
        permutation=params["permutation_int"],
        seed=params['seed_int'],
        n_cpu=params['cpu_int']).netpea_parallel()
    print("...finished DGnet.")

    print("Compute MUTnet...")
    #mutation_data = mutation_data.reset_index()
    mutation_data = pd.melt(mutation_data, id_vars=params['canc_col_name']).loc[lambda x: x["value"] > 0]
    mutation_data = mutation_data.loc[mutation_data[params['canc_col_name']].isin(response_df[params['canc_col_name']]),]
    mutation_data.iloc[:, 0:2].to_csv(params["output_dir"] + "/mutation_data.txt", sep="\t", header=True, index=False)
    MUTnet_rwr_df = rwr.RWR(
        ppiPathStr=ppi_path,
        restartPathStr=params["output_dir"] + "/drug_target.txt",
        restartProbFloat=0.5,
        convergenceFloat=0.00001,
        normalize="l1",
        weighted=True).get_prob()
    MUTnet_rwr_df.to_csv("MUTnet_rwr_df.tsv", sep='\t')
    # multiply with gene expression
    MUTnet_rwr_df = times_expression(MUTnet_rwr_df, exp_df)
    MUTnet = pea.NetPEA(
        rwrPath=MUTnet_rwr_df,
        pathwayGMT=pathway_path,
        log_transform=False,
        permutation=params["permutation_int"],
        seed=params['seed_int'],
        n_cpu=params['cpu_int']).netpea_parallel()
    print("...finished MUTnet.")    
    
    print("Compute CNVnet...")
    #cnv_data = cnv_data.reset_index()
    cnv_data = pd.melt(cnv_data, id_vars=params['canc_col_name']).loc[lambda x: x["value"] != 0]
    cnv_data = cnv_data.loc[cnv_data[params['canc_col_name']].isin(response_df[params['canc_col_name']]),]
    restart_path = params["output_dir"] + "/cnv_data.txt"
    cnv_data.iloc[:, 0:2].to_csv(params["output_dir"] + "/cnv_data.txt", sep="\t", header=True, index=False)
    CNVnet_rwr_df = rwr.RWR(
        ppiPathStr=ppi_path,
        restartPathStr=params["output_dir"] + "/drug_target.txt",
        restartProbFloat=0.5,
        convergenceFloat=0.00001,
        normalize="l1",
        weighted=True).get_prob()
    # multiply with gene expression
    CNVnet_rwr_df = times_expression(CNVnet_rwr_df, exp_df)
    CNVnet = pea.NetPEA(
        rwrPath=CNVnet_rwr_df,
        pathwayGMT=pathway_path,
        log_transform=False,
        permutation=params["permutation_int"],
        seed=params['seed_int'],
        n_cpu=params['cpu_int']).netpea_parallel()
    print("...finished CNVnet.") 

    print("run_ssgsea - compute EXP.")
    omics_data = omics.OmicsLoader(params)
    expMat = omics_data.dfs['cancer_gene_expression.tsv']
    expMat = expMat.set_index(params['canc_col_name'])
    EXP = run_ssgsea(params, expMat, response_df)

    print("prepare final input file.")
    # Read data files and rename ID columns
    #drug_mbit_df = pd.read_csv(params["drug_bits_file"], sep="\t", index_col=0)
    #drug_mbit_df = drug_mbit_df.reset_index()
    #DGnet = pd.read_csv(params["dgnet_file"], sep="\t", index_col=0)
    DGnet = DGnet.add_suffix("_dgnet").reset_index().rename(columns={"index": params['drug_col_name']})
    #CNVnet = pd.read_csv(params["cnvnet_file"], sep="\t", index_col=0)
    CNVnet = CNVnet.add_suffix("_cnvnet").reset_index().rename(columns={"index": params['canc_col_name']})
    #MUTnet = pd.read_csv(params["mutnet_file"], sep="\t", index_col=0)
    MUTnet = MUTnet.add_suffix("_mutnet").reset_index().rename(columns={"index": params['canc_col_name']})
    #EXP = pd.read_csv(params["exp_file"], sep="\t", index_col=0)
    EXP = EXP.add_suffix("_exp").reset_index().rename(columns={"index": params['canc_col_name']})
    # Extract common IDs
    common_drug_ids = reduce(np.intersect1d, (drug_mbit_df[params['drug_col_name']], DGnet[params['drug_col_name']], response_df[params['drug_col_name']]))
    common_sample_ids = reduce(np.intersect1d, (CNVnet[params['canc_col_name']],
                                                MUTnet[params['canc_col_name']],
                                                EXP[params['canc_col_name']],
                                                response_df[params['canc_col_name']]))
    # Subset to common IDs
    response_df = response_df.loc[(response_df[params['drug_col_name']].isin(common_drug_ids)) & (response_df[params['canc_col_name']].isin(common_sample_ids)), :]
    drug_mbit_df = drug_mbit_df.loc[drug_mbit_df[params['drug_col_name']].isin(common_drug_ids), :].set_index(params['drug_col_name']).sort_index()
    DGnet = DGnet.loc[DGnet[params['drug_col_name']].isin(common_drug_ids), :].set_index(params['drug_col_name']).sort_index()
    CNVnet = CNVnet.loc[CNVnet[params['canc_col_name']].isin(common_sample_ids), :].set_index(params['canc_col_name']).sort_index()
    MUTnet = MUTnet.loc[MUTnet[params['canc_col_name']].isin(common_sample_ids), :].set_index(params['canc_col_name']).sort_index()
    EXP = EXP.loc[EXP[params['canc_col_name']].isin(common_sample_ids), :].set_index(params['canc_col_name']).sort_index()
    # Join drug and sample data
    drug_data = drug_mbit_df.join(DGnet)
    sample_data = CNVnet.join([MUTnet, EXP])
    ## export train,val,test set
    for i in ["train", "test", "val"]:
        response_df = drp.DrugResponseLoader(params, split_file=params[i+"_split_file"], verbose=True)
        response_df = response_df.dfs['response.tsv']
        response_df = response_df.loc[(response_df[params['drug_col_name']].isin(common_drug_ids)) & (response_df[params['canc_col_name']].isin(common_sample_ids)),:]
        comb_data_mtx = response_df[[params['drug_col_name'], params['canc_col_name'], params['y_col_name']]]
        comb_data_mtx = (comb_data_mtx.set_index([params['drug_col_name'], params['canc_col_name'], params['y_col_name']]).join(drug_data, on=params['drug_col_name']).join(sample_data, on=params['canc_col_name']))
        ss = StandardScaler()
        comb_data_mtx.iloc[:,params["bit_int"]:comb_data_mtx.shape[1]] = ss.fit_transform(comb_data_mtx.iloc[:,params["bit_int"]:comb_data_mtx.shape[1]])
        ## add 0.01 to avoid possible inf values
        comb_data_mtx["response"] = np.log10(response_df[params['y_col_name']].values + 0.01)
        comb_data_mtx = comb_data_mtx.dropna()
        ydata = comb_data_mtx['response'].reset_index()
        frm.save_stage_ydf(ydf=ydata, stage=i, output_dir=params["output_dir"])
        pl.from_pandas(comb_data_mtx).write_csv(params["output_dir"] + "/" + frm.build_ml_data_file_name(data_format=params["data_format"], stage=i), separator="\t", has_header=True)



def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        file_path, 
        default_config="PathDSP_params.txt", 
        additional_definitions=pathdsp_preprocess_params)
    run(params)


if __name__ == "__main__":
    main(sys.argv[1:])

