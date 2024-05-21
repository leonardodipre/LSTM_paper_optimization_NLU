import torch
from torch import nn
from torch.optim import SGD
from copy import deepcopy
import math


def train_loop(args , data, optimizer, criterion, model , metrcis):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:

        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  
        
        #optimizer.step(loss.item()) # Update the weights
        if args.optimizer_name == "NTAvSGD":
            optimizer.step(loss)
        else:
            optimizer.step()

    metrcis["loss"] = sum(loss_array)/sum(number_of_tokens) 
    
        
    return sum(loss_array)/sum(number_of_tokens)



def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    
    with torch.no_grad(): 
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return



class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.1):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = m.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x

        

class NTAvSGD:
    def __init__(self, model, lr, momentum, weight_decay, nesterov=False, trigger_threshold=25):
        self.model = model
        self.optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        
        self.trigger_threshold = trigger_threshold
        self.counter = 0
        self.best_loss = float('inf')
        self.averaged_model = deepcopy(model)
        self.averaged_params = {name: param.clone() for name, param in self.averaged_model.named_parameters()}
        self.num_updates = 0

    def step(self, loss):
        self.optimizer.step()  
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        
        
        # Trigger averaging if counter exceeds the threshold
        if self.counter > self.trigger_threshold:
           
            self.num_updates += 1
            alpha = 1.0 / self.num_updates
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    self.averaged_params[name].mul_(1.0 - alpha).add_(param, alpha=alpha)
                    
                    

    def update_model_parameters(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.averaged_params[name])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_zero(self):
        self.num_updates = 0

    def get_update(self):
        print(self.num_updates)


