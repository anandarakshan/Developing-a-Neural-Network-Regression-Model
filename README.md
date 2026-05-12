# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

The experiment aims to understand the impact of different hyperparameters such as learning rate, number of hidden layers, and activation functions on model performance. It also focuses on improving the model’s generalization ability by applying techniques like normalization, regularization, and validation. Finally, the experiment helps in gaining practical knowledge of implementing deep learning models for real-world regression problems.


## Neural Network Model
<img width="953" height="567" alt="image" src="https://github.com/user-attachments/assets/6c09b6af-0eb4-43c7-8328-c6b3e14c128f" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Ananda Rakshan K V

### Register Number: 212223230014

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
  def forward(self,x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x
# Initialize the Model, Loss Function, and Optimizer
lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(lig.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      lig.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

### Dataset Information
<img width="208" height="245" alt="image" src="https://github.com/user-attachments/assets/d1f15ed0-af31-48c7-9a10-087d96afe06b" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="868" height="572" alt="image" src="https://github.com/user-attachments/assets/dff85062-d3ba-41f6-bc4e-1aa39f89f3ec" />


### New Sample Data Prediction
<img width="851" height="132" alt="image" src="https://github.com/user-attachments/assets/29d18292-cb5a-40e3-ad02-2e41898a2611" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
