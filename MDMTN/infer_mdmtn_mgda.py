from src.MDMTN_MGDA_model_MM import MDMTNmgda_MultiTaskNetwork_I
import torch
import matplotlib.pyplot as plt
from data.multi_mnist_dataloader import load_MultiMnist_data

if __name__ == '__main__':
    # Data
    data = load_MultiMnist_data()
    test_loader = data.test_dataloader()

    images, targets = next(iter(test_loader))
    label_l = targets[0][0].item()
    label_r = targets[1][0].item()

    print(f"Image batch shape: {images.shape}")
    print(f"Left label batch shape: {targets[0].shape}")
    print(f"Right label batch shape: {targets[1].shape}")
    

    # img = images[0]
    # plt.figure(figsize=(5, 5))
    # plt.imshow(img.squeeze(0), cmap='gray')
    # plt.title(f"({label_l}, {label_r})")
    # plt.axis('off')
    # plt.show(block=False)
    # plt.pause(2) 
    # plt.close()

    # Model 
    mod_params_mgda = {"batch_size": 256}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MDMTNmgda_MultiTaskNetwork_I(mod_params_mgda["batch_size"], device=device, static_a = [False, None])
    # model = torch.load("logs/MDMTN_MM_logs/MGDA_model_logs/model_states/29-03-2025--13-35-59/model_9model_task_1.pth", map_location="cpu", weights_only=False)
    model.load_model("logs/MDMTN_MM_logs/MGDA_model_logs/model_states/29-03-2025--17-20-19/model_0")
    model.eval()

    # images = images.to(device)
    # outputs = model(images.unsqueeze(-1))
    # print("Type of outputs:", type(outputs))
    # print("Output shape:", outputs.shape)
    # task1_outputs = outputs[:, :10]
    # task2_outputs = outputs[:, 10:]

    # task1_preds = task1_outputs.argmax(dim=1)
    # task2_preds = task2_outputs.argmax(dim=1)
    # for i in range(5):
    #     print(f"Image {i} | Task 1 → Pred: {task1_preds[i].item()} | GT: {targets[0][i].item()}  ||  Task 2 → Pred: {task2_preds[i].item()} | GT: {targets[1][i].item()}")
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