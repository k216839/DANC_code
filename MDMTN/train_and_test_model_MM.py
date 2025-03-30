
from load_data import load_MultiMnist_data
from src.utils.WCsAL_Train import full_training
from src.utils.WCsAL_Test import test_multitask_model
def train_and_test_model_MM(model, MultiMNISt_params):

    # Start timer
    import datetime
    from time import time
    print(datetime.datetime.now())
    t0 = time()

    train_loader, val_loader, test_loader = load_MultiMnist_data()
    
    device = MultiMNISt_params["device"]
    print(f"Training... [--- running on {device} ---]")
    
    final_model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = full_training(train_loader, val_loader, model,
                          MultiMNISt_params, init_model = True)
    
    print("Training completed !") 

    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    print("Testing ...") 
    Test_accuracy, prec_wrong_images = test_multitask_model(test_loader, final_model, MultiMNISt_params, TR_metrics) 

    return Test_accuracy, prec_wrong_images, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter
