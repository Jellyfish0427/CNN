## CNN
Implement 5-fold 
### Dataset
The Finger signs dataset has images of fingers with 6 classes (0 to 5).  
Images and labels are saved in files Dataset.zip called ‘Signs_Data_Training.h5’ and ‘Signs_Data_Testing.h5’’.  
The dataset contains 1080 and 120 images for training and testing.  
The shape of each image is 64×64.  
![image](https://user-images.githubusercontent.com/128220508/226153207-ca34db60-17ed-4434-a1a3-134768709931.png)

```js
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pylab as plt
import h5py
from torch.autograd import Variable
from sklearn import datasets
from sklearn.model_selection import KFold
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

input_shape =(-1,3,64,64) 
k_folds = 5
torch.manual_seed(42) #set fixed random number seed
```

### 1. Model architecture
```js
class My_Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      #image shape 3*64*64
      nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5), #9*60*60
      nn.BatchNorm2d(9),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2), #9*30*30
      nn.Conv2d(in_channels=9, out_channels=20, kernel_size=5), #20*26*26
      nn.BatchNorm2d(20),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2), #20*13*13

      nn.Flatten(),
      nn.Linear(20*13*13,1000),
      nn.ReLU(),
      nn.Linear(1000,500),
      nn.ReLU(),
      nn.Linear(500,100),
      nn.ReLU(),
      nn.Linear(100,6)
    )
    
  def forward(self, x):
    return self.layers(x)
```
The model is composed of layers, including convolutional, pooling, and fully connected layers.  
The convolutional layers are responsible for learning patterns in the input data, and the pooling layers are used to reduce the spatial size of the data and control overfitting.  

### 2. Hyperparameters
```js
epochs = 100
loss_function = nn.CrossEntropyLoss() #nn.CrossEntropyLoss()
lr = 0.03
batch_size = 120
model_select = My_Model()
```

### 3. Use GPU if available
```js
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
```

### 4. Read data
```js
def read_data(data_file,data_set,label_set):
  f = h5py.File(data_file,"r")
  data = f[data_set][:]
  label = f[label_set]
  data = np.float32(data)
  label = np.float32(label) 
  f.close()
  return data,label
```
```js
train_data, train_label = read_data('Signs_Data_Training.h5','train_set_x','train_set_y')
test_data, test_label = read_data('Signs_Data_Testing.h5','test_set_x','test_set_y')
```

### 5. Normalize
```js
train_data = train_data/255
test_data = test_data/255
```

### 6. Dataset and dataloader
#### Wrap train dataset
```js
class Train_Loader(Dataset):
    def __init__(self, train_data, train_label):
      self.img = train_data
      self.label = train_label
      self.data_shape = train_data.shape[0]

    def __len__(self):
      return self.data_shape

    def __getitem__(self, index):
      return self.img[index,:], self.label[index]
```

#### Wrap test dataset
```js
class Testset(Dataset):
    def __init__(self, test_data, test_label):
      self.img = test_data
      self.label = test_label
      self.data_shape = test_data.shape[0]

    def __len__(self):
      return self.data_shape

    def __getitem__(self, index):
      return self.img[index,:], self.label[index]
```

```js
train_dataset = Train_Loader(train_data, train_label)
test_dataset = Testset(test_data, test_label)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)
```

### 7. 5-fold cross validation
Cross-validation prevents overfitting by testing the model on testing data and can provide a more accurate estimate of the model's performance on new data.

#### - Some define
##### (1) Reset weights
```js
def reset_weights(m):
  #avoid weight leakage
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
```
##### (2) Calculate accuracy
```js
def cal_accuracy(prediction, label):
    ''' Calculate Accuracy, please don't modify this part
        Args:
            prediction (with dimension N): Predicted Value
            label  (with dimension N): Label
        Returns:
            accuracy:　Accuracy
    '''

    accuracy = 0
    number_of_data = len(prediction)
    for i in range(number_of_data):
        accuracy += float(prediction[i] == label[i])
    accuracy = (accuracy / number_of_data) * 100

    return accuracy
```

#### - Train model
```js
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    
    print(f'------------ FOLD {fold} ------------') #print fold

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    #data loaders in fold
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(train_dataset,batch_size, sampler=test_subsampler)
    
    cnn_model = model_select.to(device)
    cnn_model.apply(reset_weights)
    
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=lr) #initialize optimizer

    #initial accuracy array 
    training_accuracy = torch.zeros(1,epochs)
    training_accuracy = training_accuracy.to(device)

    #initial loss array 
    training_loss = torch.zeros(1,epochs)
    training_loss = training_loss.to(device)

    #training
    for epoch in range(0, epochs):
      
      if (epoch+1)%100 ==0:
        print(f'Starting epoch {epoch+1}')

      correct_train = 0
      total_train = 0
      #scheduler.step()

      #iterate training data
      for i, (images, labels) in enumerate(trainloader, 0):
        #define variables
        images = images.to(device)
        inputs = Variable(images.view(input_shape)) 
        labels = Variable(labels)
        
        #forward propagation      
        outputs = cnn_model(inputs)
        outputs = outputs.to(device)
        labels = labels.type(torch.LongTensor).to(device)
           
        #compute loss
        loss = loss_function(outputs, labels)
            
        #back propagation
        optimizer.zero_grad() #clean gradients
        loss.backward() #calculate gradients
        optimizer.step() #update parameters
        
        predicted = torch.max(outputs.data, 1)[1]
        total_train += len(labels)
        correct_train += (predicted == labels).float().sum()

        training_loss[0,epoch] += (loss / labels.size(0))
        training_accuracy[0, epoch] = 100 * correct_train / float(total_train)

    print('Training accuracy for fold %d: %f %%' % (fold, 100.0 * correct_train / float(total_train)))   
    print('Save trained model.')
```

### 8. Save the model
save_path = f'./model-fold-{fold}.pth'
torch.save(cnn_model.state_dict(), save_path)

### 9. Test the fold (Training accuracy)
```js
correct, total = 0, 0
with torch.no_grad():
  for i, (images, labels) in enumerate(testloader, 0):

    #define variables
    images = images.to(device)
    inputs = Variable(images.view(input_shape))
    labels = Variable(labels)
    labels = labels.type(torch.LongTensor).to(device)

    #forward propagation
    outputs = cnn_model(inputs).to(device)
    outputs = outputs.to(device)

    #calculate loss
    test_loss = loss_function(outputs, labels)
    predicted = torch.max(outputs.data, 1)[1]

    #total and correct
    total += len(labels) #totla numbber of labels
    correct += (predicted == labels).float().sum()

  #testing_loss[0, epoch] += (test_loss / labels.size(0))   

  # Print accuracy
  print('Accuracy for fold %d: %f %%' % (fold, 100.0 * correct / total))
  print('--------------------------------')
  results[fold] = 100 * (correct / total)
```

### 10. Plot training loss and accuracy
```js
def plot_result(epochs,plot_data,plot_label,plot_title,xlabel,ylabel):
  plt.plot(range(epochs), plot_data.T, label=plot_label, markevery=10)
  plt.title(plot_title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()
```  

```js
training_loss = training_loss.cpu().detach().numpy()
training_accuracy = training_accuracy.cpu().detach().numpy()

plot_result(epochs, training_loss, 'Training loss', 'Training loss', 'epochs', 'loss')
plot_result(epochs, training_accuracy, 'Training accuracy', 'Training accuracy', 'epochs', 'accuracy')
```

![image](https://user-images.githubusercontent.com/128220508/226156317-125878f9-be7a-4ef0-b39e-b0e61956a36e.png)  
![image](https://user-images.githubusercontent.com/128220508/226156322-30b77320-1ed3-4041-948f-1703c2b46108.png)  
By using 5- fold cross-validation, I would obtain 5 models after the training process.  
I select the model that performed the best on the validation sets, so I select the 	model of fold 0.  
The accuracy of validation data is 93.33%.  

### 11. Print fold results (Validation accuracy)
```js
print(f'Results for {k_folds} folds')
print('--------------------------------')
sum = 0.0
max_acc = 0
choose_fold = 0
for key, value in results.items():
  print(f'Fold {key}: {value} %')
  #choose max accuracy fold
  if value > max_acc:
    max_acc = value
    choose_fold = key
  sum += value
  
print(f'Average: {sum/len(results.items())} %')
print(f'Choose fold {choose_fold}')
```
![image](https://user-images.githubusercontent.com/128220508/226156107-5736754b-edec-4656-8388-ebc94ada9828.png)


### 12. Testing data accuracy
```js
for idx, (images, labels) in enumerate(test_loader):
  #define variables
  images = images.to(device)
  inputs = Variable(images.view(input_shape))
  labels = Variable(labels)
  labels = labels.type(torch.LongTensor).to(device)

  #forward propagation
  model = model_select.to(device)

  acc = test_fold(model,device,f'model-fold-{choose_fold}.pth',inputs,labels,loss_function)
  Testing_accuracy = acc
  print('Testing_accuracy(%):', Testing_accuracy)
```
Testing_accuracy(%): 89.25%

### 13. Utilize pre-trained models
Training a CNN model is time-consuming, and sometimes it is not suitable to start training the network from random initial parameters.  
I use deep learning network pre-training models in PyTorch.  
To speed up the progress of learning, I directly load the pre-training parameters in the pre-train model.  
![image](https://user-images.githubusercontent.com/128220508/226156592-6d2d2ead-2115-421a-9ed9-026394bf466c.png)  
Compare with the best fold in my model and the best model in the ResNet-18, we can see that ResNet-18 and ResNet-34 converge very fast.
