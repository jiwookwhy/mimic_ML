#Contains very simple structure needed to define GP class
#Train it and get predictions at time points we want to sample from
#these can be re-used for different features as needed

import torch
import gpytorch
import numpy as np
#am ignoring a numerical warning that we get on first couple of iter from Gpytorch...
#This should be ok as far as results go -- things seems to converge but it is annoying to see print out for every couple first iters!
import warnings
import os

#check torch and gpytorch versions
#print('Torch version: {}'.format(torch.__version__))
#print('Gpytorch version:{}'.format(gpytorch.__version__))
#mine are 1.10.2 for torch (with cuda 11.3) and 1.6 for gpytorch

#def ExactGP Class
#this is more basic layout for Regression using Gpytorch
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#for now am keeping this out of GP class (as in original nb example)
def train_GP(model, likelihood, train_x, train_y, icu_id, feat, num_iter=4000, lr=0.1):
    """
    Args
    ----
    model: instance of ExactGPModel.
    likelihood: gpytorch likelihood instance (am using GaussianLikelihood for all cases/features)
    train_x: "x" vals for training set. These are time elapased (in mins) since starttime for given ID
    train_y: "y" vals for training set. These are actual feature vals corresponding to time points in "train_x"
    num_iter: number of iterations of training to run... Defaults to 4000. This is more of less a good
    val just from eyeballing results I got for FiO2 and PEEP. Might be worth checking for other features...
    lr: learning rate. Defaults to 0.1 -- again this seems ok for cases I've seen...
    Returns
    -----
    A trained GP model which we can then use to make predictions and etc
    """
    #set model and likelihood in train mode
    model.train()
    likelihood.train()
    #get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #get mll (for loss)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    broken_covars = {}
    #train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(num_iter):
            try:
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()
                #am commeting these out
                #can get annoying when running a ton of these
                #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (i + 1, training_iter, loss.item(), \
                #model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()))
                optimizer.step()
            except:
                #save broken/failed PSD matrix
                broken_mat = output.covariance_matrix
                key = str(icu_id) + '_' + feat + '_' + str(i)
                debug_dir = './psd_mat_debug'
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                broken_covars[key] = broken_mat
                torch.save(broken_covars, os.path.join(debug_dir, key))
                pass
    return model

def get_samples(model, likelihood, df, feat_key, rate=10):
    """
    Args
    ----
    model: instance of ExactGPModel
    likelihood:gpytorch likelihood instance (am using GaussianLikelihood for all cases/features)
    feat_key: str representing feature samples refer to. Used for dict output...
    rate: sampling rate we wish to use (in min)
    Returns
    ----
    Predictions/"Samples" for timings spaced at rate given and in range of train_x
    """
    #put model and likelihood in eval mode
    model.eval()
    likelihood.eval()

    #get max time
    max_time = df['gp_times'].to_numpy()[-1]
    num_samples = int(np.rint(max_time/rate)) #this is approx...

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        vals_to_predict = torch.linspace(0, max_time, num_samples).cuda()
        pred = likelihood(model(vals_to_predict))

    #clamp predictions to enforce physiological vals (I know this sucks...)
    if feat_key=='fio2':
        pred = torch.clamp(pred.mean, min=2.99, max=100)
    else: #peep
        pred = torch.clamp(pred.mean, min=0.001, max=60) #60 would be super high
        #in original data I cut out >100 val b/c it does not seem reasonable clinically
        #and kept all else (if positive)
        #largest val (w/out 100ish measurement) was 57.

    #our sample in this case is just predicted mean
    #this is a bit diff from actually samplng (in which case we use covariance info)
    #we can do that too if you guys prefer
    #just thought mean might be better since there is lots of uncertainty for some regions in several cases I've seen
    #so we will be introducing a bunch of noise into measurements with actual samples
    return {'times':vals_to_predict.cpu().numpy(), feat_key:pred.cpu().numpy()}
