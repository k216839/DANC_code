from src.MDMTN_model_MM import SparseMonitoredMultiTaskNetwork_I
import torch
from src.utils.projectedOWL_utils import proxOWL
import matplotlib.pyplot as plt
from data.multi_mnist_dataloader import load_MultiMnist_data

if __name__ == '__main__':
    # Data
    data = load_MultiMnist_data()
    test_loader = data.test_dataloader()
    images, targets = next(iter(test_loader))
    label_l = targets[0][0].item()
    label_r = targets[1][0].item()

    # Model 
    GrOWL_parameters = {"tp": "spike", #"Dejiao", #"linear", 
                    "beta1": 0.8,  
                    "beta2": 0.2, 
                "proxOWL": proxOWL,
                "skip_layer": 1, # Skip layer with "1" neuron
                    "sim_preference": 0.7, 
                }
    GrOWL_parameters["max_layerSRate"] = 0.8
    model = SparseMonitoredMultiTaskNetwork_I(GrOWL_parameters, num_classes=[10, 10])
    model = torch.load("logs/MDMTN_MM_logs/MDMTN_model_MM_onek/model000.pth", map_location="cpu", weights_only=False)
    model.eval()

    # Predict
    plt.figure(figsize=(15, 6))
    for i in range(10):
        # Get predictions
        image = images[i].unsqueeze(0)
        outputs = model(image)
        preds = [out.argmax(dim=1).item() for out in outputs]
        
        # Get ground truth
        true_left = targets[0][i].item()
        true_right = targets[1][i].item()
        
        # Plot
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(0), cmap='gray')
        plt.title(f'Label: {true_left} {true_right} | Pred: {preds[0]} {preds[1]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()