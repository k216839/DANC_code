import numpy as np
import torch
from sklearn.cluster import AffinityPropagation

from src.utils.projectedOWL_utils import proxOWL

#######################################
#####  Helper functions for GrOWL #####
#######################################       



def similarity_info(model, zero_layers):
        with torch.no_grad():
            similarity_m = {}
            sm_p = model.GrOWL_parameters["sim_preference"]
            for name, weight in model.named_parameters():
                if (name not in zero_layers) and ('weight' in name) and (len(weight.shape)>1):
                    # Reshape the weight tensor
                    # if conv: lx(l-1)xwxh ----> (l-1)x(l.w.h)
                    # if fc: lx(l-1) ----> (l-1)xl
                    # Therefore: Weight matrix's ROWS ==> previous layer's NEURONS
                    org_shape = weight.shape
                    if ("task_blocks" in name) or (org_shape[1] == model.GrOWL_parameters["skip_layer"]): 
                        continue
                    if len(org_shape) == 2:
                        reshaped_weight = weight.T
                    else:
                        reshaped_weight = weight.view(weight.shape[1], -1)
                    
                    #n, p = reshaped_weight.shape
                    p_scl = torch.matmul(reshaped_weight, reshaped_weight.T)
                    i_norm = torch.norm(reshaped_weight, p=2, dim=1)**2
                    j_norm = i_norm.view(-1, 1)
                    sim_matrix = p_scl / torch.max(i_norm, j_norm)

                    # NaN values occur when the weight contains "zero lines"
                    # Replace them by 1.0 ====> "zero rows" are perfectly similar
                    # to belong to the same group
                    sim_matrix = torch.nan_to_num(sim_matrix, nan=1.0)
                    
                    OWL_clustering = AffinityPropagation(affinity='precomputed', preference =sm_p)
                    # OWL_clustering.fit(sim_matrix)
                    # labels = OWL_clustering.labels_
                    labels = OWL_clustering.fit_predict(sim_matrix.cpu())
                    # Get the indices of the cluster centers
                    cl_centers_idx = OWL_clustering.cluster_centers_indices_

                    # Identify zeros rows:
                    zeros_rows = torch.sum(reshaped_weight, dim=1) == 0
                    zeros_rows = zeros_rows.nonzero(as_tuple=False).squeeze().tolist()

                    similarity_m[name] = [sim_matrix, cl_centers_idx, labels, zeros_rows]
            
            return similarity_m
        #################################
            
def sparsity_info(model, verbose = True):
        sparsity_ratio = 0.0
        total_ns = 0.0
        zero_layers = []
        with torch.no_grad():
            for name, weight in model.named_parameters():
                if ('weight' in name) and (len(weight.shape)>1):
                    if len(weight.shape) == 2:
                        reshaped_weight = weight.T
                    else:   
                        reshaped_weight = weight.view(weight.shape[1], -1)
                        
                    k = torch.count_nonzero(torch.sum(reshaped_weight, dim=1) == 0).item()

                    sparsity_ratio = sparsity_ratio + k
                    total_ns = total_ns + reshaped_weight.shape[0]
                    if verbose:
                        print("Name: ", name)
                        print(f"Insignificant Neurons: {k}/{reshaped_weight.shape[0]} ({100*k/reshaped_weight.shape[0]})")
                        print("====================================")
                        
                    if torch.all(reshaped_weight == 0):
                        zero_layers.append(name)
        
        sparsity_ratio = 100*sparsity_ratio/total_ns
        print("Sparsity Ratio: ", sparsity_ratio)
        return sparsity_ratio, zero_layers
        #######################

        
def metrics_tr(model, verbose = True):
        sparsity_ratio = 0.0
        compr_ratio = 1.0
        params_shg = 1.0
        
        total_ns = 0.0
        zeros_ns = 0.0
        nonzeros_ns = 0.0
        unique_ns = 0.0
        zero_layers = []
        with torch.no_grad():
            for name, weight in model.named_parameters():
                if ('weight' in name) and (len(weight.shape)>1):
                    #print(weight)
                    if len(weight.shape) == 2:
                        reshaped_weight = weight.T
                    else:   
                        reshaped_weight = weight.view(weight.shape[1], -1)
                        
                    r = torch.count_nonzero(torch.sum(reshaped_weight, dim=1) == 0).item()
                            
                    rshpd_wght_np = reshaped_weight.cpu().numpy()
                    unique_rows = np.unique(rshpd_wght_np, axis=0)
                    
                    unique_ns = unique_ns + len(unique_rows)
                    total_ns = total_ns + reshaped_weight.shape[0]
                    nonzeros_ns = nonzeros_ns + (reshaped_weight.shape[0] - r)
                    zeros_ns = zeros_ns + r

                    if verbose:
                        print("Name: ", name)
                        print(f"Insignificant: {r}/{reshaped_weight.shape[0]} ({100*r/reshaped_weight.shape[0]} %)")
                        print("====================================")
                        
                    if torch.all(reshaped_weight == 0):
                        zero_layers.append(name)
        
        sparsity_ratio = 100*zeros_ns/total_ns
        compr_ratio = total_ns/unique_ns
        params_shg = nonzeros_ns/unique_ns
        print(" ####### Training Results ####### ")
        print("Sparsity Rate: ", sparsity_ratio)
        print("Compression Rate: ", compr_ratio)
        print("Parameter Sharing: ", params_shg)
        print(" ################################ ")
        return [sparsity_ratio, compr_ratio, params_shg], zero_layers
        ######################################
            