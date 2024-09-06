pathdsp_preprocess_params = [
    {"name": "bit_int",
     "type": int,
     "default": 128,
     "help": "Number of bits for morgan fingerprints.",
    },
    {"name": "permutation_int",
     "type": int,
     "default": 3,
     "help": "Number of permutation for calculating enrichment scores.",
    },
    {"name": "seed_int",
     "type": int,
     "default": 42,
     "help": "Random seed for random walk algorithm.",
    },
    {"name": "cpu_int",
     "type": int,
     "default": 20,
     "help": "Number of cpus to use when calculating pathway enrichment scores.",
    },    
    {"name": "drug_bits_file",
     "type": str,
     "default": "drug_mbit_df.txt",
     "help": "File name to save the drug bits file.",
    },
    {"name": "dgnet_file",
     "type": str,
     "default": "DGnet.txt",
     "help": "File name to save the drug target net file.",
    },
    {"name": "mutnet_file",
     "type": str,
     "default": "MUTnet.txt",
     "help": "File name to save the mutation net file.",
    },    
    {"name": "cnvnet_file",
     "type": str,
     "default": "CNVnet.txt",
     "help": "File name to save the CNV net file.",
    },
    {"name": "exp_file",
     "type": str,
     "default": "EXPnet.txt",
     "help": "File name to save the EXP net file.",
    },    
]

pathdsp_train_params = [
    {"name": "cuda_name",  # TODO. frm. How should we control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."
     },
    {"name": "dropout",
     "type": float,
     "default": 0.1,
     "help": "Dropout rate for the optimizer."
    },
]

pathdsp_infer_params = [
    {"name": "cuda_name",  # TODO. frm. How should we control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."
     },
]