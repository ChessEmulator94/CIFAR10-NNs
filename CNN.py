import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np

import time
import ssl
import multiprocessing

# Trains the model
def train(net, train_loader, criterion, optimizer, device):

    net.train()                                            # Set model to training mode
    running_loss = 0.0                                     # To calculate loss across the batches
                                        
    for data in train_loader:
        inputs, labels = data                              # Get input and labels for batch
        inputs = inputs.to(device)                         # Send inputs to device
        labels = labels.to(device)                         # Send labels to device
        optimizer.zero_grad()                              # Resets the gradients 
        outputs = net(inputs)                              # Get predictions
        loss    = criterion(outputs, labels)               # Calculate loss
        loss.backward()                                    # Propagate loss backwards
        optimizer.step()                                   # Update weights
        running_loss += loss.item()                        # Update loss
    return running_loss / len(train_loader)

# Tests the model
def test(net, test_loader, device):
    
    net.eval()                                             # Set model to evaluation mode
    correct = 0                                            # Measures amount of correctly classified points
    total   = 0                                            # Total amount of points tested

    with torch.no_grad():                                  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs  = inputs.to(device)                    # Send inputs to device
            labels  = labels.to(device)                    # Send labels to device
            outputs = net(inputs)                          # Get predictions
            _, predicted = torch.max(outputs.data, 1)      # Get max value
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()  # Correctly classified images
    return correct / total


# Actual CNN
class Net(nn.Module):

    def __init__(self):
        # Define convolutional layers with batch normalization
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)         # 3 input channels, 6 output channels, 5x5 kernel
        self.bn1 = nn.BatchNorm2d(6)            # 2x2 max pooling layer
        self.pool = nn.MaxPool2d(2, 2)          # 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)        
        self.bn2 = nn.BatchNorm2d(16)          
        
        # Define fully connected layers with batch normalization
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 16x5x5 input features, 120 output features
        self.bn3 = nn.BatchNorm1d(120)         
        self.fc2 = nn.Linear(120, 84)           # 120 input features, 84 output features
        self.bn4 = nn.BatchNorm1d(84)          
        self.fc3 = nn.Linear(84, 10)            # 84 input features, 10 output features

    def forward(self, x):

        # Feedforward input data through layers with batch normalization
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # Convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # Convolution -> ReLU -> Pooling
        x = x.view(-1, 16 * 5 * 5)                       # Reshape tensor
        x = F.relu(self.bn3(self.fc1(x)))                # Linear -> ReLU
        x = F.relu(self.bn4(self.fc2(x)))                # Linear -> ReLU
        x = self.fc3(x)                                 
        return x   

# __main__
def main():

    # Hyper Parameters
    BATCH_SIZE     = 32 # 64 maybe??
    NUM_WORKERS    = 2
    LEARNING_RATE  = 0.01
    MOMENTUM       = 0.9

    # Allow for test + training images to be downloaded (device dependent)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Create the training transform sequence
    train_transform = transforms.Compose([
        transforms.ToTensor(),                              # Convert to Tensor
        transforms.RandomHorizontalFlip(p=0.5),             # Randomly flip images horizontally (for training)
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  # Normalise data
    )

    # Create the testing transform sequence
    test_transform = transforms.Compose([
        transforms.ToTensor(),                              # Convert to Tensor
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  # Normalise data
    )

    # Load the training data
    trainingSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainingSet, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS)
    
    # Load the test data
    testSet    = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=NUM_WORKERS)

    # Set of classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create the model
    net = Net()

    # Classification problem uses CEL
    criterion = nn.CrossEntropyLoss()

    # Identify device
    device = ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    optimizer         = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum= MOMENTUM)
    # Allows for learning rate to be decayed at consistent intervals
    scheduler         = StepLR(optimizer, step_size=3, gamma=0.4)   
    # Allows for learning rate to be decayed during when plateaus found
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    train_start   = time.time()
    best_test     = 0
    prev_test_acc = 0
 

    if __name__ == '__main__':

        print("Hyper Params")
        print("Learning Rate: " + str(LEARNING_RATE))
        print("Momentum: " + str(MOMENTUM))
        print()

        # Train the model for 15 epochs
        for epoch in range(15):

            print("Running epoch: " + str(epoch+1))
            
            # Start time for the loop
            start_time = time.time() 
            
            train_loss = train(net, trainloader, criterion, optimizer, device)            
            test_acc = test(net, testloader, device)
            
            # Call the scheduler to update the learning rate
            scheduler.step() 
       
            end_time                             = time.time()              # End time for the loop
            loop_time                            = end_time - start_time    # Calculate loop time
            loop_time_minutes, loop_time_seconds = divmod(loop_time, 60.0)  # Get minutes and seconds
            
            print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}, Time = {loop_time_minutes:.0f}m {loop_time_seconds:.0f}s")
        
            if epoch % 4 == 0:
                            if epoch != 0:   
                                #print(test_acc)
                                #print(prev_test_acc) 
                                if test_acc - prev_test_acc <= 0.015:
                                    for param_group in optimizer.param_groups:
                                        param_group['lr'] = param_group['lr']*0.1
                                    #scheduler_plateau.step(test_acc)
                                    print("Reducing Learning Rate due to Plateau in Test Accuracy")

            if epoch % 3 == 0:
                prev_test_acc = test_acc

            if (test_acc > best_test):
                best_test = test_acc

        # End time for full training of network
        train_end = time.time()

        # Total time for network training
        total_train_time = train_end - train_start
        total_time_minutes, total_time_seconds = divmod(total_train_time, 60.0) # Get minutes and seconds

        print("Finished Training Model")
        print(f"Test accuracy = {best_test:.4f}, Time = {total_time_minutes:.0f}m {total_time_seconds:.4f}s")        
        
        # Allow for model to be saved
        modelName = input("Enter Model Name:\n")
        torch.save(net.state_dict(), modelName)
        print("Model saved as " + modelName)

if __name__ == '__main__':
    main()    
