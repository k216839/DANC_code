from src.MDMTN_MGDA_model_MM import MDMTNmgda_MultiTaskNetwork_I
import torch
import matplotlib.pyplot as plt
from data.multi_mnist_dataloader import load_MultiMnist_data

if __name__ == '__main__':
    # Data
    data = load_MultiMnist_data()
    test_loader = data.test_dataloader()
    images, targets = next(iter(test_loader))

    # Model 
    mod_params_mgda = {"batch_size": 1000} # same with batch_size of test_loader
    device = "cpu"
    model = MDMTNmgda_MultiTaskNetwork_I(mod_params_mgda["batch_size"], device=device, static_a = [False, None])
    model.load_model("logs/MDMTN_MM_logs/MGDA_model_logs/model_states/05-04-2025--19-39-50/model_0")
    model.eval()

    # Predict
    images = images.unsqueeze(-1)
    outputs = model(images)
    task1_outputs = outputs[:, :10]
    task2_outputs = outputs[:, 10:]
    
    plt.figure(figsize=(15, 6))
    for i in range(10):
        # Get predictions
        pred1 = task1_outputs[i].argmax().item()
        pred2 = task2_outputs[i].argmax().item()
        
        # Get ground truth
        true_left = targets[0][i].item()
        true_right = targets[1][i].item()
        
        # Plot
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(0), cmap='gray')
        plt.title(f'Label: {true_left} {true_right} | Pred: {pred1} {pred2}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()