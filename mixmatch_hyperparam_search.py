"""mixmatch_hyperparam_search.py Module to search hyperparameters for cityscape mixmatch"""

"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""
import optuna
from optuna.trial import TrialState

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from easydict import EasyDict

import train_mixmatch_cityscape as mix_cs
from src.models.unet import unet
from src.mixmatch import datasets as my_datasets
from src.mixmatch import losses
from src.mixmatch.mean_teacher import MeanTeacher
from src.models.misc import evaluate,evaluate_IoU
from src.mixmatch.train import train_one_epoch

from datetime import datetime
import matplotlib.pyplot as plt


def set_mixmatch_params(trial:optuna.Trial,args:EasyDict):
    
    # We change those we wish 
    # args.rampup_length = trial.suggest_int("rampup_length", 16000/args.kimg, args.epochs)
    # args.K = trial.suggest_categorical('K',[2,3])
    # args.temperature = trial.suggest_float('temperature',0,1)
    args.alpha = trial.suggest_float('alpha',0,1)
    args.lambda_u = trial.suggest_float('lambda_u',25,400)

    
    return args

def set_learning_params(trial:optuna.Trial,args:EasyDict):

    # We change those we wish 
    args.learning_rate = trial.suggest_float('lr',1e-5, 1e-1,log=True)
    args.batch_size = trial.suggest_categorical('BS',[2])
    args.lr_scheduler = trial.suggest_categorical('lr_scheduler',['OneCycleLR','CosineAnnealingLR','none'])
    
    return args

def set_weights(trial:optuna.Trial,weights:torch.Tensor):
    
    # first class always zero! (unlabeled)
    for i in range(weights.numel()-1):
        weights[i+1] = trial.suggest_float(f'{i+1}. class weight',low=1,high=10)

    return weights

def objective(trial):

    # We load the default args (specified by yaml, cmd line or default)
    args = mix_cs.parse_cmdline_arguments()

    # We set kimg and epcohs to accelerate thr training
    args.kimg = 2500
    args.epochs = 20
    args.current_count = 0
    
    # We sample and change mixmatch params
    args = set_mixmatch_params(trial,args)

    # We set learning params
    # args = set_learning_params(trial,args)

    # We set cross entropy weights
    # args.class_weights = torch.Tensor(my_datasets.CityScapeDataset.custom_class_weights).to(args.device)
    args.class_weights = set_weights(trial,args.class_weights)



    # print args
    print(f"# Starting at {datetime.now()}")
    print(f"with args:\n" + "\n".join([f"{key} : {value}" for key,value in args.items()]))

    
    
    img_size = (256,512)
    augumentation = mix_cs.prepare_transformation(img_size)
    labeled_dataloader, unlabeled_dataloader, validation_dataloader,test_dataloader =\
                                            my_datasets.get_CityScape(root=args.dataset_path,
                                                                        n_labeled = args.n_labeled,
                                                                        n_val = args.n_val,
                                                                        batch_size = args.batch_size,
                                                                        mode = 'fine', # change if you want to
                                                                        size = img_size,
                                                                        target_type='semantic',
                                                                        verbose=True
                                                                        )


    # Get network
    model = unet.Unet(args.model_architecture)
    eval_loss_fn = losses.SoftCrossEntropy(weight=args.class_weights,reduction='none')

    # Get optimizer
    optimizer  = getattr(optim,"Adam")(model.parameters(), lr=args.learning_rate)
    
    # Get scheduler
    #TODO: put it into cases:
    if args.lr_scheduler == 'OneCycleLR':
        
        lr_scheduler = getattr(optim.lr_scheduler, args.lr_scheduler)(optimizer,
                                                                      max_lr=3*args.learning_rate,
                                                                      epochs=(args.epochs-args.current_count),
                                                                      steps_per_epoch=(args.kimg // args.batch_size)
                                                                      )
    elif args.lr_scheduler == 'CosineAnnealingLR':
        epochs=(args.epochs-args.current_count)
        steps_per_epoch=(args.kimg // args.batch_size)

        lr_scheduler = getattr(optim.lr_scheduler, args.lr_scheduler)(optimizer,optimizer,epochs * steps_per_epoch)
    
    else:
        lr_scheduler = None

    # Use Teacher if desired
    if args.mean_teacher_coef:
        ewa_model = MeanTeacher(model,args.mean_teacher_coef,weight_decay=args.weight_decay)
    else:
        ewa_model = None

    writer = None
    
    metrics = EasyDict()
    metrics['train_criterion'] = np.empty(0)
    metrics['train_criterion_ewa'] = np.empty(0)
    metrics['val_loss'] = np.empty(0)
    metrics['val_acc'] = np.empty(0)
    metrics['train_loss'] = np.empty(0)
    metrics['train_acc'] = np.empty(0)
    metrics['test_loss'] = np.empty(0)
    metrics['test_acc'] = np.empty(0)
    metrics['lambda_u'] = np.empty(0)

    

    model = model.to(args.device)
    if ewa_model is not None:
        ewa_model = ewa_model.to(args.device)
    
    m = ewa_model if ewa_model is not None else model

    while args.current_count < args.epochs:

        metrics = train_one_epoch(model,m,optimizer,lr_scheduler,labeled_dataloader,unlabeled_dataloader,augumentation,args,metrics,writer)
                        
        # Compute validation metrics if validation period 
        args.current_count += 1 

        # Set evaluation metric
        tst_loss,tst_acc = evaluate(m,eval_loss_fn,test_dataloader,device=args.device)
        tst_avg_iou,class_ious = evaluate_IoU(m,test_dataloader,args.class_weights > 0)
        print(f"{args.current_count}/{args.epochs}|L: {tst_loss:4.2f}|acc: {tst_acc:4.2f}|iou: {tst_avg_iou}")
        for idx,class_iou in enumerate(class_ious):
            print(f"{idx}. class ({my_datasets.CityScapeDataset.trainid2names[idx]})\t {class_iou:4.2f}" ,end='')
        print()

        trial.report(tst_avg_iou, args.current_count)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return tst_avg_iou


if __name__ == "__main__":

    # Add stream handler of stdout to show the messages
    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    study_name = "mixmatch-hyperparam-search"  # Unique identifier of the study.
    # storage_name = "sqlite:///{}.db".format(study_name)
    # study = optuna.create_study(study_name=study_name, storage=storage_name)

    study = optuna.create_study(direction="maximize",study_name=study_name)
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    
    fig = optuna.visualization.plot_param_importances(study)
    #plt.show()
    plt.savefig('param_importances.png')
