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
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.utils as frm 
import improvelib.applications.drug_response_prediction.drp_utils as drp

from model_params_def import pathdsp_preprocess_params

file_path = Path(__file__).resolve().parent


def check_smiles_RDKit(smiles):
    smiles_to_drop = []
    for idx, row in smiles.iterrows():
        mol = Chem.MolFromSmiles(row["smile"])
        if mol is None:
            smiles_to_drop = smiles_to_drop + [idx]
    smiles = smiles.drop(smiles_to_drop)
    return smiles


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
    drug_mbit_df.to_csv(params["drug_bits_file"], header=True, index=False, sep="\t")
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
    for i in ["drug_bits_file", "dgnet_file", "mutnet_file", "cnvnet_file", "exp_file",]:
        params[i] = params["output_dir"] + "/" + params[i]
    ppi_path = params["input_supp_data_dir"] + "/STRING/9606.protein_name.links.v11.0.pkl"
    pathway_path = (params["input_supp_data_dir"] + "/MSigdb/union.c2.cp.pid.reactome.v7.2.symbols.gmt")

    print("Load drug data.")
    smiles = drp.get_x_data(file = params['drug_smiles_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])

    smiles.columns = ["smile"]

    #mutation_data = omics_data.dfs['cancer_mutation_count.tsv']
    mut = drp.get_x_data(file = params['cell_mutation_file'], 
                        benchmark_dir = params['input_dir'], 
                        column_name = params['canc_col_name'])
    cnv = drp.get_x_data(file = params['cell_cnv_file'], 
                        benchmark_dir = params['input_dir'], 
                        column_name = params['canc_col_name'])
    ge = drp.get_x_data(file = params['cell_transcriptomic_file'], 
                        benchmark_dir = params['input_dir'], 
                        column_name = params['canc_col_name'])
    
    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    smiles = check_smiles_RDKit(smiles)

    drug_info = pd.read_csv(params["input_dir"] + "/x_data/drug_info.tsv", sep="\t")
    drug_info["NAME"] = drug_info["NAME"].str.upper()
    target_info = pd.read_csv(params["input_supp_data_dir"] + "/data/DB.Drug.Target.txt", sep="\t")
    target_info = target_info.rename(columns={"drug": "NAME"})
    targets = pd.merge(drug_info, target_info, how="left", on="NAME").dropna(subset=["gene"])
    targets = targets[[params['drug_col_name'], 'gene']]
    targets.set_index(params['drug_col_name'])



    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        response_stage = drp.get_response_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                response_file=params['y_data_file'])
        print("1", response_stage.shape)
        response_stage = drp.get_response_with_features(response_stage, [ge, mut, cnv], params['canc_col_name'])
        print("2", response_stage.shape)
        response_stage = drp.get_response_with_features(response_stage, [smiles, targets], params['drug_col_name'])
        print("3", response_stage.shape)
        ge_stage = drp.get_features_in_response(ge, response_stage, params['canc_col_name'])
        mut_stage = drp.get_features_in_response(mut, response_stage, params['canc_col_name'])
        cnv_stage = drp.get_features_in_response(cnv, response_stage, params['canc_col_name'])
        smiles_stage = drp.get_features_in_response(smiles, response_stage, params['drug_col_name'])
        targets_stage = drp.get_features_in_response(targets, response_stage, params['drug_col_name'])

        print("Convert drug to bits...")
        drug_mbit_df = smile2bits(params, smiles_stage)
        print("...finished drug to bits.")

        print("Compute DGnet...")
        print("DGnet - prep data...")

        #combined_df = combined_df.loc[combined_df[params['drug_col_name']].isin(response_all[params['drug_col_name']]),]
        targets_stage.to_csv(params["output_dir"] + "/drug_target.txt", sep="\t", header=True, index=False)
        print("DGnet - random walk with restart...")
        DGnet_rwr_df = rwr.RWR(
            ppiPathStr=ppi_path,
            restartPathStr=params["output_dir"] + "/drug_target.txt",
            restartProbFloat=0.5,
            convergenceFloat=0.00001,
            normalize="l1",
            weighted=True).get_prob()
        print("DGnet - NetPEA...")
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
        print("MUTnet - prep data...")
        mut_stage = mut_stage.reset_index()
        mut_stage = pd.melt(mut_stage, id_vars=params['canc_col_name']).loc[lambda x: x["value"] > 0]
        mut_stage.iloc[:, 0:2].to_csv(params["output_dir"] + "/mutation_data.txt", sep="\t", header=True, index=False)
        print("MUTnet - random walk with restart...")
        MUTnet_rwr_df = rwr.RWR(
            ppiPathStr=ppi_path,
            restartPathStr=params["output_dir"] + "/mutation_data.txt",
            restartProbFloat=0.5,
            convergenceFloat=0.00001,
            normalize="l1",
            weighted=True).get_prob()
        MUTnet_rwr_df.to_csv("MUTnet_rwr_df.tsv", sep='\t')
        # multiply with gene expression
        print("MUTnet - multiply by expression...")
        MUTnet_rwr_df = times_expression(MUTnet_rwr_df, ge_stage)
        print("MUTnet - NetPEA...")
        MUTnet = pea.NetPEA(
            rwrPath=MUTnet_rwr_df,
            pathwayGMT=pathway_path,
            log_transform=False,
            permutation=params["permutation_int"],
            seed=params['seed_int'],
            n_cpu=params['cpu_int']).netpea_parallel()
        print("...finished MUTnet.")    
        
        print("Compute CNVnet...")
        cnv_stage = cnv_stage.reset_index()
        cnv_stage = pd.melt(cnv_stage, id_vars=params['canc_col_name']).loc[lambda x: x["value"] != 0]
        cnv_stage.iloc[:, 0:2].to_csv(params["output_dir"] + "/cnv_data.txt", sep="\t", header=True, index=False)
        CNVnet_rwr_df = rwr.RWR(
            ppiPathStr=ppi_path,
            restartPathStr=params["output_dir"] + "/cnv_data.txt",
            restartProbFloat=0.5,
            convergenceFloat=0.00001,
            normalize="l1",
            weighted=True).get_prob()
        # multiply with gene expression
        CNVnet_rwr_df = times_expression(CNVnet_rwr_df, ge_stage)
        CNVnet = pea.NetPEA(
            rwrPath=CNVnet_rwr_df,
            pathwayGMT=pathway_path,
            log_transform=False,
            permutation=params["permutation_int"],
            seed=params['seed_int'],
            n_cpu=params['cpu_int']).netpea_parallel()
        print("...finished CNVnet.") 

        print("run_ssgsea - compute EXP.")
        EXP = run_ssgsea(params, ge_stage, response_stage)

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
        print("length of drug_mbit_df:", len(drug_mbit_df[params['drug_col_name']]))
        print("length of DGnet:", len(DGnet[params['drug_col_name']]))
        print("length of response_df:", len(response_stage[params['drug_col_name']]))
        print("length of unique drug_mbit_df:", len(drug_mbit_df[params['drug_col_name']].unique()))
        print("length of unique DGnet:", len(DGnet[params['drug_col_name']].unique()))
        print("length of unique response_df:", len(response_stage[params['drug_col_name']].unique()))
        common_drug_ids = reduce(np.intersect1d, (drug_mbit_df[params['drug_col_name']], DGnet[params['drug_col_name']], response_stage[params['drug_col_name']]))
        print("length of CNVnet:", len(CNVnet[params['canc_col_name']]))
        print("length of MUTnet:", len(MUTnet[params['canc_col_name']]))
        print("length of EXP:", len(EXP[params['canc_col_name']]))
        print("length of response_df:", len(response_stage[params['canc_col_name']]))
        print("length of unique CNVnet:", len(CNVnet[params['canc_col_name']].unique()))
        print("length of unique MUTnet:", len(MUTnet[params['canc_col_name']].unique()))
        print("length of unique EXP:", len(EXP[params['canc_col_name']].unique()))
        print("length of unique response_df:", len(response_stage[params['canc_col_name']].unique()))
        common_sample_ids = reduce(np.intersect1d, (CNVnet[params['canc_col_name']],
                                                    MUTnet[params['canc_col_name']],
                                                    EXP[params['canc_col_name']],
                                                    response_stage[params['canc_col_name']]))
        # Subset to common IDs
        print("response before subset shape:", response_stage.shape)
        response_stage = response_stage.loc[(response_stage[params['drug_col_name']].isin(common_drug_ids)) & (response_stage[params['canc_col_name']].isin(common_sample_ids)), :]
        print("response after subset shape:", response_stage.shape)
        drug_mbit_df = drug_mbit_df.loc[drug_mbit_df[params['drug_col_name']].isin(common_drug_ids), :].set_index(params['drug_col_name']).sort_index()
        DGnet = DGnet.loc[DGnet[params['drug_col_name']].isin(common_drug_ids), :].set_index(params['drug_col_name']).sort_index()
        CNVnet = CNVnet.loc[CNVnet[params['canc_col_name']].isin(common_sample_ids), :].set_index(params['canc_col_name']).sort_index()
        MUTnet = MUTnet.loc[MUTnet[params['canc_col_name']].isin(common_sample_ids), :].set_index(params['canc_col_name']).sort_index()
        EXP = EXP.loc[EXP[params['canc_col_name']].isin(common_sample_ids), :].set_index(params['canc_col_name']).sort_index()
        # Join drug and sample data
        drug_data = drug_mbit_df.join(DGnet)
        sample_data = CNVnet.join([MUTnet, EXP])
        ## export train,val,test set


        response_stage = response_stage.loc[(response_stage[params['drug_col_name']].isin(common_drug_ids)) & (response_stage[params['canc_col_name']].isin(common_sample_ids)),:]
        comb_data_mtx = response_stage[[params['drug_col_name'], params['canc_col_name'], params['y_col_name']]]
        comb_data_mtx = (comb_data_mtx.set_index([params['drug_col_name'], params['canc_col_name'], params['y_col_name']]).join(drug_data, on=params['drug_col_name']).join(sample_data, on=params['canc_col_name']))
        ss = StandardScaler() ## need to fix this
        comb_data_mtx.iloc[:,params["bit_int"]:comb_data_mtx.shape[1]] = ss.fit_transform(comb_data_mtx.iloc[:,params["bit_int"]:comb_data_mtx.shape[1]])
        ## add 0.01 to avoid possible inf values
        comb_data_mtx["response"] = np.log10(response_stage[params['y_col_name']].values + 0.01)
        comb_data_mtx = comb_data_mtx.dropna()
        ydata = comb_data_mtx['response'].reset_index()
        frm.save_stage_ydf(ydf=ydata, stage=stage, output_dir=params["output_dir"])
        pl.from_pandas(comb_data_mtx).write_csv(params["output_dir"] + "/" + frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage), separator="\t", has_header=True)



def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        file_path, 
        default_config="PathDSP_params.txt", 
        additional_definitions=pathdsp_preprocess_params)
    run(params)


if __name__ == "__main__":
    main(sys.argv[1:])

