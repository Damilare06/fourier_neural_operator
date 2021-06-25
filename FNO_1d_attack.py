""" 
    Simple attack showing the Burgers 1d model attack for different epsolons vs accuracy
    Run with "python3.8 single_attack.py"
"""

import torch
import torchvision.models as models
import torch.nn.functional as F
from fourier_1d_module import * 
from utilities3 import *

def mse_attack(model, X, y, epsilon=0.1, alpha=1e-5, num_iter=1):
    """ 
        Construct PGD-like MSE attacks on X with minimal default values"
    """
    delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = F.mse_loss(model(X + delta).squeeze(), y)
        loss.backward()
        delta.data = (delta + X.shape[0] * alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def mse_linf_attack(model, X, y, epsilon=0.02, alpha=1e-3, num_iter=40):
    """ 
        Construct PGD-like MSE attacks on X"
    """
    delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = F.mse_loss(model(X + delta).squeeze(), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def mse_linf_rand_attack(model, X, y, epsilon, alpha, num_iter, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X.squeeze())
    """ 
        Construct PGD-like MSE attacks on X"
    """
    for i in range(restarts):
        delta = torch.rand_like(X.squeeze(), requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon

        for t in range(num_iter):
            loss = F.mse_loss(model(X + delta).squeeze(), y)
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()

        all_loss = F.mse_loss(model(X + delta).squeeze(), y, reduction='none')            
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta


def epoch_adversarial(model, loader, attack, attack_name, *args):
    total_loss = 0.
    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        delta = attack(model, X, y.squeeze(), *args)
        yp = model(X + delta)
        loss = F.mse_loss(yp.squeeze(), y.squeeze())
        
        total_loss += loss

    output =  total_loss / len(loader.dataset)
    print(f"{attack_name} error: {output :.6f} ")
    return output

def show_burgers_overlap(var1, var2, key1, key2):
    cm = plt.cm.get_cmap('viridis')
    var = [var1.numpy(), var2.numpy()]
    var = np.asarray(var)
    var = var.reshape(var.shape[1], var.shape[2], var.shape[0])
    print(var.shape)
    key = [key1, key2]

    # Generate the data distribution
    fig, ax = plt.subplots()
    ax.set_title(f"N = {var.shape[0]} of {key[0]} vs {key[1]} on {var.shape[1]} grid\n")
    var_hist = var.reshape(var.shape[0] * var.shape[1], var.shape[2])
    print(var_hist.shape)
    ax.hist(var_hist, density=True, stacked=False, bins='auto', label=key)
    ax.set_xlabel(f"{key} distribution")
    ax.legend()

def show_burgers(var, key):
    cm = plt.cm.get_cmap('viridis')
    var = var.numpy()

    # Generate the data distribution
    fig, ax = plt.subplots()
    ax.set_title(f"N = {var.shape[0]} of {key} on {var.shape[1]} grid\n")
    var_hist = var.flatten()
    n, bins, patches = ax.hist(var_hist, density=True, stacked=True, bins='auto', color="green")
    ax.set_xlabel(f"{key} distribution")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))    
    plt.show()

def main() -> None:
    #  configurations
    ntest = 100
    sub = 2**3 # subsampling rate
    h = 2**13 // sub # total grid size divided by the subsampling rate
    s = h

    batch_size = 1 # 20
    debug = False 

    # read data
    # ... of the shape (number of samples, grid size)
    dataloader = MatReader('data/burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]

    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    # cat the locations information
    grid = np.linspace(0, 1, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)

    # load and initialize the model
    model = torch.load('model/ns_fourier_burgers').eval()

    # Evaluation - the foolbox accuracy loss is for classification
    pred = torch.zeros(y_test.shape)
    index = 0
    test_mse = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            pred[index] = out.squeeze()

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')            
            test_mse += mse.item()

            index = index + 1
    test_mse /= ntest
    # show_burgers(y_test, 'y_test')
    # show_burgers(pred, 'y_pred')
    show_burgers_overlap(y_test, pred, 'y_test', 'y_pred')
    plt.show()
    print(f"test_mse loss before attack: {test_mse :.6f} ")

    # Comment: alpha = step size, epsilon = perturbation range 
    eps = 0.1
    num_iter = 10
    restarts = 10
    alpha = eps/ num_iter

    # epoch_adversarial(model, test_loader, mse_attack, "mse_attack", eps, alpha, num_iter)
    # epoch_adversarial(model, test_loader, mse_linf_rand_attack, "mse_linf_rand_attack", eps, alpha, num_iter, restarts)
    # epoch_adversarial(model, test_loader, mse_linf_attack, "mse_linf_attack", eps, alpha, num_iter)



if __name__ == "__main__":
    main()