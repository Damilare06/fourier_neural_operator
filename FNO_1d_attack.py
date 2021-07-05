""" 
    Simple attack showing the Burgers 1d model attack for different epsolons vs accuracy
    Run with "python3.8 single_attack.py"
"""

import torch
from torch.serialization import load
import torchvision.models as models
import torch.nn.functional as F
from fourier_attack_module import * 
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


def epoch_adversarial(model, loader, attack, attack_name, xtest, *args):
    total_loss = 0.
    index = 0
    a_plus_delta = torch.zeros_like(xtest)
    delta_arr = torch.zeros_like(xtest)

    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        delta = attack(model, X, y.squeeze(), *args)
        if 'rand' in attack_name:
            delta[:,1] = 0
        else:
            delta[:,:,1] = 0
        yp = model(X + delta)
        loss = F.mse_loss(yp.squeeze(), y.squeeze())
        
        total_loss += loss
        a_plus_delta[index,:,:] = X + delta
        delta_arr[index,:,:] = delta

        index += 1

    output =  total_loss / len(loader.dataset)
    print(f"{attack_name} error: {output :.6f} ")

    return delta_arr[:,:,0].squeeze(), a_plus_delta[:,:,0].squeeze()

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
    evaluate = False

    # read data of the shape (number of samples, grid size)
    # dataloader = MatReader('data/burgers_data_R10.mat')
    if evaluate:
        dataloader = MatReader('data/burgers_N2048_G8192.mat')
        x_data_full = dataloader.read_field('a')[:,:]
        x_test = x_data_full[-ntest:,::sub]
        y_test = dataloader.read_field('u')[-ntest:,::sub]

        # cat the locations information
        grid = np.linspace(0, 1, s).reshape(1, s, 1)
        grid = torch.tensor(grid, dtype=torch.float)
        x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)

        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)

        # load and initialize the model
        # model = torch.load('model/ns_fourier_burgers').eval()
        model = torch.load('model/ns_N2048_G8192_burgers').eval()

        # Evaluation - the foolbox accuracy loss is for classification
        pred = torch.zeros(y_test.shape)
        index = 0
        test_mse = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).squeeze()
                pred[index] = out

                mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')            
                test_mse += mse.item()

                index = index + 1
        test_mse /= ntest
        # show_burgers(y_test, 'y_test')
        # show_burgers(pred, 'y_pred')
        # show_burgers_overlap(y_test, pred, 'y_test', 'y_pred')
        # plt.show()
        print(f"test_mse loss before attack: {test_mse :.9f} ")

        # Comment: alpha = step size, epsilon = perturbation range 
        eps = 0.1
        num_iter = 10
        restarts = 10
        alpha = eps/ num_iter
        # delta_arr = torch.zeros_like(x_test)

        # Get delta, a + delta
        delta_out, a_plus_delta = epoch_adversarial(model, test_loader, mse_attack, "mse_attack", x_test, eps, alpha, num_iter)

        # epoch_adversarial(model, test_loader, mse_linf_attack, "mse_linf_attack", x_test, eps, alpha, num_iter)
        # epoch_adversarial(model, test_loader, mse_linf_rand_attack, "mse_linf_rand_attack", x_test, eps, alpha, num_iter, restarts)

        # POST-PROCESSING
        a_p_delta = x_data_full[-ntest:,:]
        delta = torch.zeros_like(x_data_full[:ntest,:])

        a_p_delta[:,::sub] = a_plus_delta
        delta[:,::sub] = delta_out
        # print("ABJ: ", a_p_delta.shape, delta.shape)

        scipy.io.savemat('pred/a_p_delta_burger_N2048_G8092.mat', mdict={'a_plus_delta': a_p_delta.cpu().numpy(),'delta': delta.cpu().numpy(), 'y_pred': pred.cpu().numpy()})

    # Read both files, get y_pred and u from second file


    ##################################################################
    # second part, compare the MSE of y_pred and u (from the gen file)
    ##################################################################
    dataloader_1 = MatReader('pred/a_p_delta_burger_N2048_G8092.mat')
    y_pred = dataloader_1.read_field('y_pred')[:,:]
    dataloader_2 = MatReader('data/burgers_N2048_G8192_gen.mat')
    u_delta = dataloader_2.read_field('u')[:, ::sub]

    print(y_pred.shape)
    print(u_delta.shape)

    total_mse = 0
    for i, yp in enumerate(y_pred):
        mse = F.mse_loss(yp.view(1, -1), u_delta[i].view(1, -1), reduction='mean')            
        total_mse += mse.item()

    total_mse /= ntest
    print (total_mse)

if __name__ == "__main__":
    main()