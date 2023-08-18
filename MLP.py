import torch
import torchvision
import torchvision.transforms as transforms                 # Image Augmentation
import torch.nn as nn  
import torch.nn.functional as F                             # Layers
import torch.optim as optim                                 # Optimizers
import matplotlib.pyplot as plt
import time
import ssl
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


# MLP Structure
class MLP(nn.Module):

    # Defines the netowrk's layers 
    def __init__(self):
        
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()         # Flattens the 2D image for input into first layer
        self.fc1 = nn.Linear(32*32*3, 512)  
        self.bn1 = nn.BatchNorm1d(512)      # Batch Normalization after first layer
        self.fc2 = nn.Linear(512, 256)      # First Hidden Layer
        self.bn2 = nn.BatchNorm1d(256)      # Batch Normalization after second layer
        self.fc3 = nn.Linear(256, 128)      # Second Hidden Layer
        self.bn3 = nn.BatchNorm1d(128)      # Batch Normalization after third layer
        self.fc4 = nn.Linear(128, 10)       # Third Hidden Layer
        self.output = nn.LogSoftmax(dim=1)  

    # Defines the forward pass
    def forward(self, x):
        
        x = self.flatten(x)                 # Batch now has shape (B, C*W*H)
        x = F.relu(self.fc1(x))             # First Hidden Layer
        x = self.bn1(x)                     # Batch Normalization after first layer
        x = F.relu(self.fc2(x))             # Second Hidden Layer
        x = self.bn2(x)                     # Batch Normalization after second layer
        x = F.relu(self.fc3(x))             # Third Hidden Layer
        x = self.bn3(x)                     # Batch Normalization after third layer
        x = self.fc4(x)                     # Output Layer
        x = self.output(x)                  

        return x  
    

# Training Function
def train(net, train_loader, criterion, optimizer, device):

    net.train()                              # Set model to training mode.
    running_loss = 0.0                       # To calculate loss across the batches
    
    # Iterate through all batches in the training data
    for data in train_loader:

        # Get input and labels for batch
        inputs, labels = data             
        
        # Send inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)  
        
        # Zero out the gradients of the network i.e. reset
        optimizer.zero_grad()             
        
        # Get predictions and calculate loss
        outputs = net(inputs)             
        loss = criterion(outputs, labels) 
        
        # Propagate loss backwards using automatic differentiation
        loss.backward()  
        
        # Update weights and running loss
        optimizer.step()                  
        running_loss += loss.item()       
        
    # Return average loss across all batches in the training data
    return running_loss / len(train_loader)


# Test Function
def test(net, test_loader, device):
    net.eval()                               # Set model to evaluation mode
    correct = 0                              # Measures amount of correctly classified points
    total = 0                                # Total amount of points tested
    
    with torch.no_grad(): 
        # Iterate through all test data
        for data in test_loader:
            # Get inputs and labels for batch
            inputs, labels = data
            
            # Send inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device) 
            
            # Get predictions and calculate accuracy
            outputs = net(inputs) 
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
            
    # Return the accuracy of the model on the test data
    return correct / total


def main():

    # Hyper Parameters 
    LEARNING_RATE = 0.016 
    MOMENTUM      = 0.9
    BATCH_SIZE    = 128

    # Allow for test + training images to be downloaded (device dependent)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Create the training transform sequence
    train_transform = transforms.Compose([
        transforms.ToTensor(),                              # Convert to Tensor
        transforms.RandomHorizontalFlip(p=0.5),             # Randomly flip images horizontaly (for training)
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  # Normalise data
    )

    # Create the testing transform sequence
    test_transform = transforms.Compose([
        transforms.ToTensor(),                              # Convert to Tensor
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  # Normalise data
    )

    # Get training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    # Get test set
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)

    # Send data to the data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True)

    test_loader  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False)

    # Identify device
    device = ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Create the model and send its parameters to the appropriate device
    mlp = MLP().to(device)
    
    # Define the loss function, optimizer, and learning rate scheduler
    criterion         = nn.NLLLoss()
    optimizer         = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler         = StepLR(optimizer, step_size=3, gamma=0.4)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    train_start = time.time()
    best_test = 0
    prev_test_acc = 0.4


    print("Hyper Params")
    print("Learning Rate: " + str(LEARNING_RATE))
    print("Momentum: " + str(MOMENTUM))
    print()

    # Perform training + testing
    for epoch in range(15):

        print("Running epoch: " + str(epoch+1))

        # Start time for the loop
        start_time = time.time() 

        train_loss = train(mlp, train_loader, criterion, optimizer, device)
        test_acc   = test(mlp, test_loader, device)
        
        # Call the scheduler to update the learning rate
        scheduler.step()             
        
        # Get time values
        end_time                             = time.time()              # End time for the loop
        loop_time                            = end_time - start_time    # Calculate loop time
        loop_time_minutes, loop_time_seconds = divmod(loop_time, 60.0)  # Get minutes and seconds

        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}, Time = {loop_time_minutes:.0f}m {loop_time_seconds:.0f}s")

        # Check if model has hit a plateau
        # Implement learning rate decay if TRUE
        # Uses accuracy from (3) epochs ago to test for plateau
        if epoch % 4 == 0:
                if epoch != 0:   
                    # print(test_acc)
                    # print(prev_test_acc) 
                    if test_acc - prev_test_acc <= 0.015:
                        scheduler_plateau.step(test_acc)
                       # print("Reducing Learning Rate due to Plateau in Test Accuracy")

        # Lagging comparison accuracy to test for plateauing 
        if epoch % 3 == 0:
            prev_test_acc = test_acc

        # Store best accuracy
        if (test_acc > best_test):
            best_test = test_acc

    # Get final end time for network training
    train_end = time.time()

    total_train_time = train_end - train_start
    total_time_minutes, total_time_seconds = divmod(total_train_time, 60.0) # Get minutes and seconds

    print("Finished Training Model")
    print(f"Best test accuracy = {best_test:.4f}, Time = {total_time_minutes:.0f}m {total_time_seconds:.4f}s")

    # Allow for model to be saved
    modelName = input("Enter Model Name:\n")
    torch.save(mlp.state_dict(), modelName)
    print("Model saved as " + modelName)


if __name__ == '__main__':
    main()
