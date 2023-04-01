import torch
from torch.autograd import Variable

class linearRegression(torch.nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__() 
        self.lr_pytorch = torch.nn.Linear(1,1)

    def forward(self,input_to_predict):
        predictions = self.lr_pytorch(input_to_predict)
        return predictions
            
    def fit(self,inputs,outputs,epochs = 500, learning_rate = 0.01):
        #inputs = Variable(torch.Tensor(inputs))
        #labels = Variable(torch.Tensor(outputs))
        
        stochastic_gradient_descent = torch.optim.SGD(lr_pytorch_model.parameters(),learning_rate)
        mse_loss_function = torch.nn.MSELoss() 
        
        for epoch in range(epochs):
            stochastic_gradient_descent.zero_grad()
            predictions = lr_pytorch_model(inputs)
            
            loss = mse_loss_function(predictions,labels)
            loss.backward()
            stochastic_gradient_descent.step()
        
