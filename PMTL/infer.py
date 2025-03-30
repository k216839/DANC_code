import pickle
import torch
import numpy as np 
import matplotlib.pyplot as plt
from model_lenet import RegressionModel, RegressionTrain
with open('MTL_dataset/multi_mnist.pickle','rb') as f:
    trainX, trainLabel,testX, testLabel = pickle.load(f)  

trainX = torch.from_numpy(trainX.reshape(120000,1,36,36)).float()
trainLabel = torch.from_numpy(trainLabel).long()
testX = torch.from_numpy(testX.reshape(20000,1,36,36)).float()
testLabel = torch.from_numpy(testLabel).long()
train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
test_set  = torch.utils.data.TensorDataset(testX, testLabel) 
 
batch_size = 256
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader))) 
state_dict = torch.load('logs/model_mtl.pickle', map_location='cpu')
n_tasks = 2
init_weight = np.array([0.5 , 0.5 ])
model = RegressionModel(n_tasks)
model.load_state_dict(state_dict)
model = RegressionTrain(model, init_weight)
if torch.cuda.is_available():
    model.cuda()

total_test_loss = []
test_acc = []

correct1_test = 0
correct2_test = 0

for (X, ts) in test_loader:
    if torch.cuda.is_available():
        X = X.cuda()
        ts = ts.cuda()

    valid_test_loss = model(X, ts)
    total_test_loss.append(valid_test_loss)
    output1 = model.model(X).max(2, keepdim=True)[1][:,0]
    output2 = model.model(X).max(2, keepdim=True)[1][:,1]
    correct1_test += output1.eq(ts[:,0].view_as(output1)).sum().item()
    correct2_test += output2.eq(ts[:,1].view_as(output2)).sum().item()
     
test_acc = np.stack([1.0 * correct1_test / len(test_loader.dataset),1.0 * correct2_test / len(test_loader.dataset)])

total_test_loss = torch.stack(total_test_loss)
average_test_loss = torch.mean(total_test_loss, dim = 0)
print('==>>> test acc:')
print(test_acc)
print('==>>> average test loss:')
print(average_test_loss)

X_batch, ts_batch = next(iter(test_loader))
if torch.cuda.is_available():
    X_batch = X_batch.cuda()
    ts_batch = ts_batch.cuda()

# Dự đoán của mô hình
output = model.model(X_batch)  # Model dự đoán
output1 = output.max(2, keepdim=True)[1][:, 0]  # Kết quả task 1
output2 = output.max(2, keepdim=True)[1][:, 1]  # Kết quả task 2

# Chuyển tensor về CPU để vẽ
X_batch = X_batch.cpu()
ts_batch = ts_batch.cpu()
output1 = output1.cpu()
output2 = output2.cpu()

# Hiển thị 10 ảnh đầu tiên
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    if i >= 10:
        break
    img = X_batch[i].squeeze() 
    
    ax.imshow(img, cmap='gray')
    ax.axis("off")
    
    true_label = f"{ts_batch[i,0].item()} {ts_batch[i,1].item()}"
    pred_label = f"{output1[i].item()} {output2[i].item()}"
    ax.set_title(f"GT: {true_label} | Pred: {pred_label}")

plt.tight_layout()
plt.show()
