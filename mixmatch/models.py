"""models.py: Module containing models (Pytorch nn.modules) of plausible classifiers
This is module which contains different architecture of classifiers.
"""
import torch
import torch.nn as nn
import copy
from itertools import chain

class MLP(nn.Module):
    """a ANN representing a multilayer preceptron (fully connected/dense/linear layers)"""

    def __init__(self,layers:list,activation_function:nn.Module=nn.ReLU(),name="MLP"):
        """
        layers : list containing size of input and output dimensions. Length of the list corresponds to the number
            of layers 
        activation_function : nn.Module arbitrary non-linear activation function for MLP
        """
        super(MLP, self).__init__()

        self.num_classes = layers[-1]
        self.activation = activation_function
        self.name = name

        body_list = list(chain.from_iterable( 
                                (nn.Linear(idim,odim),
                                self.activation)  for idim,odim in zip(layers[:-2],layers[1:-1])))
        
        idim,odim = (layers[-2],self.num_classes)
        body_list.append(nn.Linear(idim,odim))            
        self.body = nn.Sequential(*body_list)
        
    def forward(self,x):
        scores = self.body(x)

        return scores
    


class Mean_Teacher(nn.Module):
    def __init__(self, student_model:nn.Module,alpha:float=0.995):
        """Mean Teacher model implementation of
        
        arxiv.org/abs/1703.01780
        https://github.com/CuriousAI/mean-teacher

        Args:
            student_model (nn.Module): Student model instance (used for initialization)
            alpha (float): ewa weight (used for updating the model weights)
        """
        super().__init__()
        self.model = copy.deepcopy(student_model)
        self.alpha = alpha
        self.step = 1
        for param in self.model.parameters():
            param.requires_grad = False

    
    def forward(self,x):
        return self.model(x)

    def update_weights(self,student_model:nn.Module):
        """https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py line 189"""
        alpha = self.alpha
        alpha = min(1 - 1 / (self.step + 1), alpha)
        self.step += 1
        for teacher_param,student_param in zip(self.model.parameters(),student_model.parameters()):
            teacher_param.data.mul_(self.alpha).add_(student_param * (1 - alpha))


