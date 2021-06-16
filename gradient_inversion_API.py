import sys
import logging
import torch
import datetime
import inversefed
import numpy as np
from scipy.optimize import linear_sum_assignment
import piq

### EARLY STOPPING CLASS FOR FASTER RECONSTRUCTION
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Code inspired by https://github.com/Bjarten/early-stopping-pytorch 26.03.2021
    """
    def __init__(self, patience=7, delta=0, metric='loss', subject_to='min', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            metric (str): string of the metric we are looking at in our history
            subject_to (str): Defines whether the metric is subject to minimazation or maximization; 'min' or 'max' (defaut or when misspelled: 'min')
            verbose (bool): If True, logs a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.metric = metric
        self.subject_to = subject_to
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.improved = False
        self.delta = delta

    def get_state(self):
        return self.__dict__

    def set_state(self, state_dict):
        self.__dict__ = state_dict

    def __call__(self, metric):
        if self.subject_to == 'max':
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.improved = True
            self.best_score = score
        elif score >= self.best_score + self.delta:
            self.improved = False
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.improved = True
            self.best_score = score
            self.counter = 0

### MSE Metric Class
class Metric:
    def __init__(self):
        self.metric_fn = None
    def __call__(self, prediction, truth):
        if self.metric_fn == None:
            raise  NotImplementedError()
        else:
            return self.metric_fn(prediction, truth)
            
class MSE(Metric):
    def __init__(self, reduce=True):
        if reduce:
            self.metric_fn = torch.nn.MSELoss(size_average=None, reduction='mean')
        else:
            self.metric_fn = self.mse
        self.format = '.6f'
        self.reduce = reduce
        self.name = 'MSE'
        self.target = 'features'

    def mse(self, x, y):
        if not self.reduce:
            mse_fn = torch.nn.MSELoss(size_average=None, reduction='none')
            value = mse_fn(x, y)
            for _ in range(len(value.shape)-1):
                value = value.mean(dim=-1)
            return value
        else:
            print('You shouldn\'t be here.')

class SSIM(Metric):
    def __init__(self, reduce = True):
        self.metric_fn = self.ssim
        if reduce:
            self.reduction = 'mean'
        else:
            self.reduction = 'none'
        self.format = '.5f'
        self.name = 'SSIM'
        self.target = 'features'
    def ssim(self, img_batch, ref_batch):
        return piq.ssim(normalize_tensor_01(img_batch), normalize_tensor_01(ref_batch), reduction = self.reduction)


def normalize_tensor_01(tensor, batch=True):
    if batch:
        t = []
        for x in tensor:
            t.append(_normalize_single_tensor_01(x))
        return torch.stack(t)
    else:
        return _normalize_single_tensor_01(tensor)

def _normalize_single_tensor_01(tensor):
    return (tensor.max()-tensor) / (tensor.max() - tensor.min())


### DEFINE SOME BASIC FUNCTIONS
def system_startup(args=None, defs=None):
    """Set Logging"""
    rootLogger = logging.getLogger()
    logFormatter = logging.Formatter('%(asctime)s:[%(levelname)s][%(filename)s][%(funcName)s] %(message)s')
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

    """Log useful system information."""
    # Choose GPU device and print status information:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logging.info('Currently evaluating -------------------------------:')
    logging.info(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    logging.info(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()}.')
    if args is not None:
        logging.info(args)
    if defs is not None:
        logging.info(repr(defs))
    if torch.cuda.is_available():
        logging.info(f'GPU : {torch.cuda.get_device_name(device=device)}')
    return device
    

def get_gradient(model, input_data, gt_labels, loss_fn, train_mode):
    print('Generatig gradient from victim data...')
    if train_mode:
        model.train()
    else:
        model.eval()

    model.zero_grad()
    target_loss = loss_fn(model(input_data), gt_labels)
    gradient = torch.autograd.grad(target_loss, model.parameters(), allow_unused=True)
    #return gradient
    gradient = [grad.detach() for grad in gradient]

    return gradient

def gradient_inversion(gradient, labels, model, data_shape, dm, ds):
    print('Performing a Gradientinversion attack.')
    #build inversefed library specific config for the reconstruction attack
    c = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=7000,
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss',
              loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'), #(Loss fn the model was trianed with)
              early_stopper = EarlyStopping(1000, 0, 'ReconstructionLoss', 'min', False)
              )
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), c, num_images=labels.shape[0])
    output, stats = rec_machine.reconstruct(gradient, labels, img_shape=data_shape)

    return output, stats['opt']


def match_reconstructions(images, reconstructions, metric='ssim'):
    cost_matrix = get_similarity_cost_matrix(images, reconstructions, metric)
    rec_idx = linear_sum_assignment(cost_matrix, maximize=True)[1]
    return reconstructions[rec_idx].detach().clone()


def get_similarity_cost_matrix(images, reconstructions, metric='ssim'):
    if metric == 'mse':
        m = MSE(False)
    else: 
        m = SSIM(False)
    cost_matrix = []
    for img in images:
        i, r = torch.broadcast_tensors(img, reconstructions)
        current_metric = m(i, r)
        cost_matrix.append(np.array(current_metric).astype(float))
    return np.array(cost_matrix)


