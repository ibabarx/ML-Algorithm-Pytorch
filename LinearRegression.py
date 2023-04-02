import torch
from torch.autograd import Variable
import numpy as np

class linearRegression(torch.nn.Module):            
    
    # instantiating Linear Regression class
    def __init__(self):
        super(linearRegression, self).__init__() 
        self.lr_pytorch = torch.nn.Linear(1,1)                                                       # Linear block in pytorch

    def forward(self,input_to_predict):                                                              # Forward propagation (Prediction)
        predictions = self.lr_pytorch(input_to_predict)
        return predictions
            
    def fit(self,inputs,outputs,epochs = 10, learning_rate = 0.01):                                 # Fit method to train on existing data
        
        inputs = Variable(torch.from_numpy(np.array(inputs,dtype=np.float32).reshape(-1,1)))        # converting np.array to tensors for operations
        labels = Variable(torch.from_numpy(np.array(outputs,dtype=np.float32).reshape(-1,1))) 
        
        stochastic_gradient_descent = torch.optim.SGD(self.lr_pytorch.parameters(),learning_rate)  # instantiating SGD Gradient Descent 
        mse_loss_function = torch.nn.MSELoss()                                                      # Instantiationg MSE Loss function
        
        for epoch in range(epochs):                                                                 # Training 
            stochastic_gradient_descent.zero_grad()
            predictions = self.lr_pytorch(inputs)
            
            loss = mse_loss_function(predictions,labels)
            loss.backward()
            stochastic_gradient_descent.step()
