"""mean_teacher.py:  Module containing mean teacher (rather EWA of modelsweights)"""
import torch
import torch.nn as nn
import copy
from itertools import chain

class MeanTeacher(nn.Module):
    def __init__(self, student_model:nn.Module,alpha:float=0.999,weight_decay:float=0.0004):
        """Mean Teacher model implementation of
        
        arxiv.org/abs/1703.01780
        https://github.com/CuriousAI/mean-teacher

        Args:
            student_model (nn.Module): Student model instance (used for initialization)
            alpha (float): ewa weight (used for updating the model weights)
            weight_decay(float): weight decay applied on the student model instance 
        """
        super().__init__()
        self.model = copy.deepcopy(student_model)
        self.alpha = alpha
        self.step = 1
        self.wd  = weight_decay
        for param in self.model.parameters():
            param.requires_grad = False
            param.detach_()

        
    def forward(self,x):
        return self.model(x)

    def update_weights(self,student_model:nn.Module):
        """Motivated by https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py line 189"""
        alpha = self.alpha
    
        self.step += 1
        model_params = list(self.model.state_dict().values())
        student_params = list(student_model.state_dict().values())
        for teacher_param,student_param in zip(model_params,student_params): # not working with model.parameters() method
            if teacher_param.dtype == torch.float32:
                teacher_param.mul_(self.alpha).add_( (1 - alpha) * student_param)
                # customized weight decay (https://github.com/google-research/mixmatch/blob/master/mixmatch.py#L92)
                student_param.mul_(1-self.wd)