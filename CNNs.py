
#>>>
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




#>>>
from sklearn.model_selection import train_test_split
train_ratio= 0.8

x_train, x_test, y_train, y_test= train_test_split(data, mark, train_size= train_ratio)
x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train, train_size= train_ratio)






