import torch
import torch.optim as optim
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from src.resnet_model import MiniResNet
from src.validation import validate
from torch.utils.data import Subset
from src.gradient_hook import get_gradient_hook
from src.cnn_model import CIFARR10CNN



#1. Define transform ( Turn images into tensors )
transform = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip()])



#2. Download/Load datasets
train_dataset = torchvision.datasets.CIFAR10(root="./../experiments/data",train=True,download=True, transform=transform)


test_dataset = torchvision.datasets.CIFAR10(root="./../experiments/data",train=False,download=True, transform=transform)

# 2.5 Create a subset of 1000 images
subset_indices = list(range(10000))
train_subset = Subset(train_dataset, subset_indices)
test_subset = Subset(test_dataset,subset_indices)


#3. Create loaders
train_loader = DataLoader(train_subset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_subset,batch_size=64,shuffle=False)


#4. Sanity check
data_iter = iter(train_loader)
images,labels = next(data_iter)



#5.Device check
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


print(f"Batch Images Shape: {images.shape}")
#(batch_size, channels, height,width)

print(f"Batch Labels Shape: {labels.shape}")


model = CIFARR10CNN()

##Gradient hook : Register hook for specific layers
gradients = {}
#We register them to the main blocks to track the highway
model.fc2.register_full_backward_hook(get_gradient_hook("6_fc2",gradients))
model.fc1.register_full_backward_hook(get_gradient_hook("5_fc1",gradients))
model.conv4.register_full_backward_hook(get_gradient_hook("4_conv4",gradients))
model.conv3.register_full_backward_hook(get_gradient_hook("3_conv3",gradients))
model.conv2.register_full_backward_hook(get_gradient_hook("2_conv2",gradients))
model.conv1.register_full_backward_hook(get_gradient_hook("1_conv1",gradients))

criterion = torch.nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)


#Training Loop
epochs = 15
best_accuracy = 0.0
history = []
checkpoint_dir = "experiments/run_latest"


for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    best_val_loss = float('inf')
    
    
    for i,(images,labels) in enumerate(train_loader):
        #Zero gradient
        optimizer.zero_grad()
        
        
        #Forward Pass
        outputs = model(images)
        
        loss = criterion(outputs,labels)
        
        
        
        #Backward Pass + update
        loss.backward()
        
        if(i == 0):
            print(f"Gradients at Epoch {epoch+1} :{gradients}")
            
            
        if(1%10==0):   
            print(f"This is the loss:{loss.item()} at this iteration: {i} at epoch: {epoch+1}")
        
        optimizer.step()
        
        
        #Metrics Calculation
        running_loss += loss.item()
        
        
        _,predicted = torch.max(outputs,1)
        
        total += labels.size(0)
        
        
        correct += (predicted == labels).sum().item()
        
        
    #Print stats per epoch
    train_accuracy = 100 * correct/total
    
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} -Training Accuracy: {train_accuracy:.2f}%")
    
    
    avg_loss, accuracy = validate(model, test_loader,criterion,device)
    
    
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Test Accuracy: {accuracy:.2f}%")
    
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        
        
        
    epoch_metrics = {
        "epoch": epoch + 1,
        "train_loss": running_loss /len(train_loader),
        "train_acc": accuracy,
    }
    
    
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        
        
    state = {
        "epoch":epoch_metrics["epoch"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "lr":0.01,
            "dropout": 0.5,
            "weight_decay": 1e-4
        }
    }
    
    
    history.append(epoch_metrics)
    
    
    if (epoch + 1) % 5 == 0 or avg_loss < best_val_loss:
        pass
    
with open("./experiments/training_plaincnn_log.json","w") as f:
    json.dump(history, f, indent=4)
    
    
    

    
    
    