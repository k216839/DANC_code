import torch
import torch.optim as optim
from src.utils.projectedOWL_utils import proxOWL
from src.MDMTN_model_MM import SparseMonitoredMultiTaskNetwork_I
from src.MDMTN_MGDA_model_MM import MDMTNmgda_MultiTaskNetwork_I

def get_params(k, archi_name, main_dir, mod_logdir, num_model, Sparsity_study = True):
    mod_params = {"w": k, "a": torch.zeros(3),
                  "epsilon": 0.0001, "num_tasks": 3, "num_outs": [10, 10],
                  "max_iter": 12, "max_iter_search": 3, "max_iter_retrain": 10, 
                        "lr": 0.005, "lr_sched_coef": 0.5, "LR_scheduler": True, 
                    "num_epochs": 3, "num_epochs_search": 3, "num_epochs_retrain": 3, "tol_epochs": None,
                    "num_model": num_model,"main_dir": main_dir, "mod_logdir": mod_logdir,
                    "mu": 2.5e-08,  
                    "rho": 2, 
                    "base_optimizer": optim.Adam, "is_search": Sparsity_study, "Sparsity_study": Sparsity_study,
                    "criterion": torch.nn.functional.nll_loss,}
        
    GrOWL_parameters = {"tp": "spike", #"Dejiao", #"linear", 
                "beta1": 0.8,  
                "beta2": 0.2, 
            "proxOWL": proxOWL,
            "skip_layer": 1, # Skip layer with "1" neuron
                "sim_preference": 0.7, 
            }
    
    if archi_name.lower() == "mdmtn":
        mod_params["min_sparsRate"] = 20.00 # (20 %)
        GrOWL_parameters["max_layerSRate"] = 0.8 # (80 %)
        model = SparseMonitoredMultiTaskNetwork_I(GrOWL_parameters, mod_params["num_outs"], static_a = [False, None])
    else: raise ValueError(f"Unknown model architecture {archi_name} !")

    return model, mod_params, GrOWL_parameters

def get_params_mgda(archi_name, model_dir_path, device):
    mod_params_mgda = { "lr": 1e-2, "momentum": 0.9,
                    "model_repetitions": 1, "training_epochs": 100,
                    "archi": archi_name,"img_shp": (28, 28, 1), "model_dir_path": model_dir_path,
                    "batch_size": 256}
    
    if archi_name.lower() == "mdmtn":
        model = MDMTNmgda_MultiTaskNetwork_I(mod_params_mgda["batch_size"], device=device, static_a = [False, None])
    else: raise ValueError(f"Unknown model architecture {archi_name} !")
    return model, mod_params_mgda