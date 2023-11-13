"""
Short script containing both scikit and torch models we use as our
base clfs in the FQI_functions module.
"""

#import stuff we need
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np

#sklearn clfs
#these are sparse tree ensembles we used
#and MLPs (which am keeping here for now) but we do NOT use formally for our MLP experiments...
#see torch version of MLPs (which we used instead) below

def fitTree(samples,targets):
    clf = ExtraTreesRegressor(n_estimators=50, max_depth=None, min_samples_leaf=50, random_state=1, warm_start=False, n_jobs=12)
    clf.fit(samples, targets)
    return clf

def fitCtree(states, actions):
    clf = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_leaf=50, class_weight='balanced', n_jobs=12)
    clf.fit(states, actions)
    return clf

def fitNN(samples, targets):
    ## might need to amp up n_iters used here
    ## for scikit think default max_iters == 200
    #so we prolly want to do more than 200 epochs
    clf = MLPRegressor(solver='adam', alpha=1e-5,learning_rate='adaptive',hidden_layer_sizes=(5,2),random_state=None)
    clf.fit(samples, targets)
    return clf

def fitCNN(samples, targets):
    clf = MLPClassifier(solver='adam', alpha=1e-5,learning_rate='adaptive',hidden_layer_sizes=(5,2),random_state=None)
    clf.fit(samples, targets)
    return clf

def partialfitNN(clf, samples, targets):
    clf.partial_fit(samples, targets)
    return clf

#Torch MLPs

#vanilla MLP
class MLPTorch(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer_sizes):
        """
        Basic two-layer multilayer perceptron regressor.

        Args:
            in_dim: number of input features, should be 37 for us (35
                    measurements + 2 actions)
            hidden_layer_sizes: tuple of number of units per hidden layer.
        """
        super(MLPTorch, self).__init__()
        assert len(hidden_layer_sizes) > 0

        #setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #setup layers
        self.fc_layers = nn.ModuleList()
        # input -> first hidden layer
        self.fc_layers.append(nn.Linear(in_dim, hidden_layer_sizes[0]))
        # hidden layer -> hidden layer
        for prev_size, next_size in zip(hidden_layer_sizes, hidden_layer_sizes[1:]):
            self.fc_layers.append(nn.Linear(prev_size, next_size))
        # hidden layer -> output
        self.fc_layers.append(nn.Linear(hidden_layer_sizes[-1], out_dim))

        #put layers in device
        self.fc_layers.to(self.device)

    def forward(self, x):
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))

        x = self.fc_layers[-1](x)

        return x

    def predict(self, x):
        """
        x is assumed to be a numpy array [n_obs, n_features]
        """
        with torch.no_grad():
            x = torch.FloatTensor(x).to(self.device)
            return self.forward(x).cpu().numpy()

#attention-MLP (basic version)
#for now am doing this as separate class (and re-using training method!)
class AttnMLPTorch(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer_sizes):
        """
        Two-layer multilayer perceptron regressor with attention mechanism.

        Args:
            in_dim: number of input features, should be 37 for us (35
                    measurements + 2 actions)
            hidden_layer_sizes: tuple of number of units per hidden layer.
        """
        super(AttnMLPTorch, self).__init__()
        assert len(hidden_layer_sizes) > 0

        #setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #setup layers
        self.fc_layers = nn.ModuleList()
        #input -> linear layer for attention
        #this is simple linear layer with output size == in_dim
        self.fc_layers.append(nn.Linear(in_dim, in_dim))
        # attn output --> actual first hidden layer
        self.fc_layers.append(nn.Linear(in_dim, hidden_layer_sizes[0]))
        # hidden layer -> hidden layer
        for prev_size, next_size in zip(hidden_layer_sizes, hidden_layer_sizes[1:]):
            self.fc_layers.append(nn.Linear(prev_size, next_size))
        # hidden layer -> output
        self.fc_layers.append(nn.Linear(hidden_layer_sizes[-1], out_dim))

        self.fc_layers.to(self.device)

    def forward(self, x):

        #apply attn mech here
        h = self.fc_layers[0](x) #feed thourh linear layer
        h = F.softmax(h, dim=1) #pass res through softmax
        h = h * x #element-wise mult w/ original input

        #run res through regular MLP
        for fc_layer in self.fc_layers[1:-1]:
            h = F.relu(fc_layer(h))

        h = self.fc_layers[-1](h)

        return h

    def predict(self, x):
        """
        x is assumed to be a numpy array [n_obs, n_features]
        """
        with torch.no_grad():
            x = torch.FloatTensor(x).to(self.device)
            return self.forward(x).cpu().numpy()


def fitTNN(samples, targets, target_type, model=None, epochs=30, mlp_type='vanilla', lr=1e-5):
    # for now use the argument "target_type" to select between discrete vs continuous targets
    # set model to None if train from scratch
    #mlp type refers to which MLP version/class to use -- can be either 'vanilla' or 'attn'
    if target_type == 'continuous':
        criterion = nn.MSELoss()
    elif target_type == 'discrete':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("target_type must be either 'continuous' or 'discrete'")

    # (1) convert samples and targets to tensors and batch
    X = torch.FloatTensor(np.array(samples))
    y = torch.FloatTensor(np.array(targets))
    dataset = TensorDataset(X, y)
    in_dim = X.shape[1]
    out_dim = 1 if len(y.shape) == 1 else y.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_layer_sizes = (5, 2)

    if mlp_type=='vanilla':
        mlp = model if model else MLPTorch(in_dim, out_dim, hidden_layer_sizes).to(device) # create model if DNE
    elif mlp_type=='attn':
        mlp = model if model else AttnMLPTorch(in_dim, out_dim, hidden_layer_sizes).to(device)
    else:
        raise NotImplementedError("mlp_type must be either 'vanilla' or 'attn'")

    # Use 200 as batchsize like sklearn default
    loader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=0)

    # (2) train for some epochs
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    for _ in range(epochs):
        for feats, targs in loader:
            feats = feats.to(device)
            targs = targs.to(device)
            optimizer.zero_grad()
            out = mlp(feats)
            targs = targs.view(targs.size(0), -1)
            loss = criterion(out, targs)
            loss.backward()
            optimizer.step()
    return mlp
