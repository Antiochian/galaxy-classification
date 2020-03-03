pyt# -*- coding: utf-8 -*- python3
"""
Created on Sun Mar  1 00:07:53 2020

@author: Antiochian
"""
from __future__ import print_function, division
import os
import scipy.ndimage
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import math
from skimage import io
# different recognised types of galaxies
global classes
classes = ['spiral', 'elliptical', 'uncertain']

# detect CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)

#NETWORK PARAMETERS
# helper function to compute the output dimension after a convolution
def get_conv_out_dim(size, padding, dilation, kernel, stride):
    return int(np.floor((size+2*padding-dilation*(kernel-1)-1)/stride + 1))
w0 = 120 #size of first layer
w1 = get_conv_out_dim(w0, 0, 1, 8, 2)
w2 = get_conv_out_dim(w1, 0, 1, 2, 2)
w3 = get_conv_out_dim(w2, 0, 1, 8, 2)
w4 = get_conv_out_dim(w3, 0, 1, 2, 2)

class Classifyer_CNN(nn.Module):
    def __init__(self):
        super(Classifyer_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 8, (2, 2))  # in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode
        self.conv2 = nn.Conv2d(6, 16, 8, (2, 2))
        
        self.act1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2, 2)        
        
        self.act2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * w4 * w4, 120)
        self.act3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(120, 60)  
        self.act4 = nn.LeakyReLU()
        self.fc3 = nn.Linear(60, 3) #FINAL output layer
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * w4 * w4) #reshape into 16 rows to feed into 16-high second conv layer
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)  # dont take softmax yet -s done automatically later by the loss function
        return x

def make_neural_network():
    print("Building NN...")
    net = Classifyer_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.to(device)
    return criterion,optimizer,net

#read data
def read_data(datapath='./data/'):
    hnd = open(datapath+"training_data.txt", "r")
    all_data = eval(hnd.read())
    hnd.close()
    return all_data

def get_mean_and_variance(all_data,datapath='./data/'):
    """Get average pixel value to help centre and normalise dataset"""
    image_list = [io.imread(datapath + item[0] + '.jpg') for item in all_data]
    mean = np.mean(image_list)/255 #normalize these
    std_dev = np.std(image_list)/255
    std_dev = 0.5
    print("mean: ",mean,"std dev: ",std_dev)
    return mean, std_dev

def over_under_sample(all_data, transform, inverse_transform,datapath='./data/',split_fraction=0.9):
    """This function takes in all the data, and over/under samples each category 
    to populate a test/training set with a roughly even spread of categories.
    Oversampling is done with rotations due to the symettry of the problem"""
    
    global classes
    num_of_images = len(all_data)
    counts = [0 for _ in range(len(classes))] #count how many of each type are in dataset
    for i in range(num_of_images):
        category_info = all_data[i][1:]
        if sum(category_info) != 1:
            raise Exception("Undefined category for object:",all_data[i])
        cat_index = category_info.index(1)
        counts[cat_index] += 1
    
    target_balance = 4*min(counts) #goal no. of each class to aim for
    
    balanced_image_data = []
    newcounts = [0 for _ in range(len(classes))] #no. of each class taken so far (starts [0,0,0])
    for i in range(num_of_images):
        #copy all images to balanced_ list if its not already full to capacity
        c = all_data[i][1:].index(1)
        diff = target_balance - newcounts[c] #how many left to supply 
        if diff > 0: #if not yet full
            new_images = []
            category_tensor = torch.tensor(all_data[i][1:]).to(device)
            img0 = plt.imread(os.path.join(datapath, all_data[i][0] + '.jpg'))
            new_image_tensor= transform(np.array(img0)).to(device)
            balanced_image_data.append([new_image_tensor, category_tensor]) #reuse same category data
            newcounts[c] += 1
    for repeats in range(3): #3 more possible rotations, innit
        for i in range(num_of_images):
            #ADD CURRENT IMAGE TO DATASET
            c = all_data[i][1:].index(1)
            diff = target_balance - newcounts[c] #how many left to supply     
            if diff > 0: #if not yet full
                new_images = []
                category_tensor = torch.tensor(all_data[i][1:]).to(device)
                img0 = plt.imread(os.path.join(datapath, all_data[i][0] + '.jpg'))
                new_images.append(img0)
                if diff >= 2:
                    img1 = scipy.ndimage.rotate(img0, 90)
                    new_images.append(img1)
                    if diff >= 3:
                        img2 = scipy.ndimage.rotate(img0, 180)
                        new_images.append(img2)
                        if diff >= 4:
                            img3 = scipy.ndimage.rotate(img0, 270)
                            new_images.append(img3)
                for img in new_images:
                    new_image_tensor= transform(np.array(img)).to(device)
                    balanced_image_data.append([new_image_tensor, category_tensor]) #reuse same category data
                    newcounts[c] += 1
    
    random.shuffle(balanced_image_data)  #randomise order of dataset  
    split_point = math.floor(len(balanced_image_data)*split_fraction)
    train_set = balanced_image_data[:split_point]
    test_set = balanced_image_data[split_point:]
        
    print("(# train, # test): (" + str(len(train_set)) + ", " + str(len(test_set)) + ")\n")
    
    for i in range(len(classes)):
        print(classes[i]," : ", newcounts[i]/float(len(train_set)))
    return train_set, test_set
    
def train(criterion, optimizer, net, train_set, test_set, num_epochs=15, batch_size=1,threshold=0.7):
    trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    num_of_batches = len(train_set)//batch_size
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # implement training loop here, as in Problem Set 1.
        running_loss = 0.0
        print("-"*20)
        print("Epoch", epoch)
        print("-"*20)
        for i, data in enumerate(trainloader, 0):
    
            inputs, labels = data
    
            # zero the parameter gradients, note we do this after each input (SGD)
            optimizer.zero_grad()
    
            # forward, backward, optimize
            outputs = net(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if not i % 499:    # print every 500 batches
                print("\rCurrent Batch: ",i+1,"/",num_of_batches,"\tCurrent Loss:",round(running_loss/500,5),end='')
                running_loss = 0.0
        print("\rBatches Complete: ",num_of_batches,"/",num_of_batches,"\t  Final Loss:",round(running_loss/500,5))
        with torch.no_grad():
            num_correct = [0, 0, 0]
            num_in_category = [0, 0, 0]
            incorrect_classified = []
            for _, data in enumerate(testloader, 0):
                #structure goes like this:
                # iter = [id, data] ; data = [tensor_input, predicted_labels]
                inputs, labels = data
                labels = torch.max(labels, 1)[1]
                outputs = net(inputs)
                outputs = torch.max(outputs, 1)[1] #choose most confident label
                num_in_category[labels] += 1
                if labels == outputs:
                    num_correct[outputs] += 1
                else:
                    incorrect_classified.append([inputs[0], classes[labels], classes[outputs]])
            
            print("\nACCURACY DATA:")
            for i in range(len(num_correct)):
                print("{:12s}: {:1.4f}".format(classes[i], float(num_correct[i])/float(num_in_category[i])))
                print("{:12s}: {:3d}/{:3d} = {:1.4f}".format("Total Acc.", len(test_set)-len(incorrect_classified), len(test_set), float(len(test_set)-len(incorrect_classified)) / float(len(test_set))))

def isolated_test(test_set,net, batch_size=1):
    testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
            num_correct = [0, 0, 0]
            num_in_category = [0, 0, 0]
            incorrect_classified = []
            for _, data in enumerate(testloader, 0):
                #structure goes like this:
                # iter = [id, data] ; data = [tensor_input, predicted_labels]
                inputs, labels = data
                labels = torch.max(labels, 1)[1]
                outputs = net(inputs)
                outputs = torch.max(outputs, 1)[1] #choose most confident label
                num_in_category[labels] += 1
                if labels == outputs:
                    num_correct[outputs] += 1
                else:
                    incorrect_classified.append([inputs[0], classes[labels], classes[outputs]])
            
            print("\nACCURACY DATA:")
            for i in range(len(num_correct)):
                print("{:12s}: {:1.4f}".format(classes[i], float(num_correct[i])/float(num_in_category[i])))
                print("{:12s}: {:3d}/{:3d} = {:1.4f}".format("Total Acc.", len(test_set)-len(incorrect_classified), len(test_set), float(len(test_set)-len(incorrect_classified)) / float(len(test_set))))
    return

def main(datapath='./data/'):
    print("Reading data...")
    all_data = read_data(datapath)
    mean, std_dev = get_mean_and_variance(all_data)
    print("\tDone.")

    #define these handy transforms to normalise the data    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean,mean,mean),(std_dev,std_dev,std_dev))]) #another shaky assumption about average colors here
    inverse_transform = transforms.Compose([transforms.Normalize((-mean/std_dev,-mean/std_dev,-mean/std_dev), (1/std_dev,1/std_dev,1/std_dev)),transforms.ToPILImage()])
    
    print("Over/Undersampling data (may take a while)...")
    train_set, test_set = over_under_sample(all_data, transform, inverse_transform, datapath)
    print("\tDone.")
    print("Making neural network...")
    criterion, optimizer, net = make_neural_network()
    print("\tDone.")
    print("Training neural network...")
    train(criterion, optimizer, net, train_set, test_set)                
    print("\tDone.")
    choice = input("Save model to folder? (Y/N): ")
    if choice == "Y" or choice == "y":
         torch.save(net.state_dict(), "model.pth")
         print("File saved to 'model.pth'")
    return net
   
if __name__ == "__main__":
	main()