
#>>>>>>>>>>>>>>>>
'''
Input Dimensions:
For a 2D convolution (Conv2D), the input data typically has the following dimensions:

1-Batch size: The number of samples (images) in a batch.
2-Channels: The number of channels in the image (e.g., 3 for RGB, 1 for grayscale).
3-Height: The height (number of rows) of the image.
4-Width: The width (number of columns) of the image.

'''
data= data.unsqueeze(1).permute(0,1,3,2)
torch.save(data, 'data.pth')
torch.save(mark, 'mark.pth')
print(data.shape)
print(mark.shape)









#>>>>>>>>>>>>>>>>
from sklearn.model_selection import train_test_split
train_ratio= 0.8

x_train, x_test, y_train, y_test= train_test_split(data, mark, train_size= train_ratio)
x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train, train_size= train_ratio)
print('train: ', x_train.shape, y_train.shape)
print('valid: ', x_valid.shape, y_valid.shape)
print('test: ', x_test.shape, y_test.shape)














#>>>>>>>>>>>>>>>>
#batching  
from torch.utils.data import DataLoader, TensorDataset
train_batch_size= 330
valid_batch_size= 330

train_dataset= TensorDataset(x_train, y_train)
valid_dataset= TensorDataset(x_valid, y_valid)
test_dataset= TensorDataset(x_test, y_test)

train_loader= DataLoader(train_dataset, batch_size= train_batch_size, shuffle= True)
valid_loader= DataLoader(valid_dataset, batch_size= valid_batch_size, shuffle= False)
test_loader= DataLoader(test_dataset, batch_size= valid_batch_size, shuffle= False)

print("train batch size:",train_loader.batch_size, ", num of batch:", len(train_loader))
print("valid batch size:",valid_loader.batch_size, ", num of batch:", len(valid_loader))
print("test batch size:",test_loader.batch_size, ", num of batch:", len(test_loader))


x,y= next(iter(train_loader))
print(x.shape, y.shape)
#output: torch.Size([330, 1, 22, 200]) torch.Size([330])









#>>>>>>>>>>>>>>>>
#How Using nn.Conv2d
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
self.conv2d= nn.Conv2d(num_input, F1, kernel_size_1, padding=kernel_padding_1)

'''
Parameters:
in_channels: The number of input channels (e.g., 3 for RGB images, 1 for grayscale). depth of in and out. 
out_channels: The number of output channels (i.e., the number of filters to apply).
kernel_size: The size of the convolutional filter (e.g., 3 for a 3x3 kernel, 5 for a 5x5 kernel). This can be an integer or a tuple (height, width).
stride (optional): The step size of the filter as it slides over the input. Default is 1.
padding (optional): The number of zero-padding added to the input on each side. Default is 0.
Padding ensures that the output has the same spatial dimensions as the input (if used correctly).
dilation (optional): The spacing between kernel elements. Default is 1.
groups (optional): Controls connections between inputs and outputs. Default is 1, meaning every input is connected to every output.  >>> Depthwise_conv2D :  Groups depth layers.
bias (optional): If True, adds a learnable bias to the output. Default is True.
'''

import torch
import torch.nn as nn
import torch.optim as optim
# Define a simple Conv2D layer with:
# 1 input channel (e.g., grayscale image), 3 output channels (3 filters), and a 3x3 kernel
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
input_data = torch.randn(2, 1, 5, 5)  # 2 images, 1 channel, 5x5 size
output = conv_layer(input_data)
print(output.shape)  # Output shape

#>
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define two Conv2D layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Output: [batch_size, 16, 28, 28]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Output: [batch_size, 32, 28, 28]
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2x

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply Conv1, ReLU, and Max Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply Conv2, ReLU, and Max Pooling
        return x

# Create a model instance
model = SimpleCNN()

# Example input: 1 grayscale image of size 28x28 (e.g., MNIST image)
input_image = torch.randn(1, 1, 28, 28)  # 1 image, 1 channel, 28x28 size

# Get output from the model
output = model(input_image)

print(output.shape)  # Output shape
#<











#>>>>>>>>>>>>>>>>
self.Batch_normalization_1 = nn.BatchNorm2d(F1)

'''
BatchNorm2d is the 2D batch normalization layer, commonly used after convolutional layers.
F1 is the number of feature maps (channels) that will be normalized. This value should match the number of channels in the output of the previous layer.
For example, if you have 32 output channels from a Conv2d layer, you would set F1 = 32.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Define convolutional layer with 3 input channels and 16 output channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # Define Batch Normalization layer for 16 feature maps (output channels of conv1)
        self.Batch_normalization_1 = nn.BatchNorm2d(16)
        
        # Define another convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Define Batch Normalization for 32 feature maps (output channels of conv2)
        self.Batch_normalization_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        # Apply conv1 -> BatchNorm1 -> ReLU
        x = self.Batch_normalization_1(F.relu(self.conv1(x)))
        
        # Apply conv2 -> BatchNorm2 -> ReLU
        x = self.Batch_normalization_2(F.relu(self.conv2(x)))
        
        return x

# Example input: 8 RGB images of size 32x32
input_image = torch.randn(8, 3, 32, 32)

# Create a model instance
model = SimpleCNN()

# Get output from the model
output = model(input_image)

print(output.shape)  # Output shape after convolution and batch normalization












#>>>>>>>>>>>>>>>>



















