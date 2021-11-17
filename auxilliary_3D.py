import torch
from torch.serialization import load
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

def pgd_linf(model, loader, attack, attack_name, xtest, *args):
    total_loss = 0.
    index = 0
    a_plus_delta = torch.zeros_like(xtest)
    delta_arr = torch.zeros_like(xtest)

    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        delta = attack(model, X, y.squeeze(), *args)
        # print(X.shape, y.shape, delta.shape)

        # zero out the loc_delta index
        delta[:,:,:,:,:3] = 0

        yp = model(X + delta)
        loss = F.mse_loss(yp.squeeze(), y.squeeze())
        
        total_loss += loss
        a_plus_delta[index,:,:,:,:] = X + delta
        delta_arr[index,:,:,:,:] = delta

        index += 1
    output =  total_loss / len(loader.dataset)
    print(f"The proxy {attack_name} error => MSE(model(a + delta), model(a)) = : {output :.6f} ")

    return delta_arr[:,:,:,:,:], a_plus_delta

def norms(Z):
    """ Compute the norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None,None]

def pgd_l2(model, X, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        loss = F.mse_loss(yp.squeeze(), y.squeeze())
        loss.backward()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        #delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()        
    
    delta = delta.detach()
    delta[:,:,:,:,:3] = 0
    a_plus_delta = X + delta
    # delta = delta[:,:,0].squeeze()
    return delta.cpu(), a_plus_delta.cpu()


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

# ground_truth_mse = MSE(model(a+delta), solver(a+delta))
def get_ground_truth_mse(model, ap_delta, file, var, ntest, sub):
    dataloader = MatReader(file)
    u_delta = dataloader.read_field(var)[:, ::sub].cuda()

    # iterate through ap_delta
    total_mse = 0
    for i in range(ntest):
        apd = ap_delta[i, :, :]
        apd = torch.unsqueeze(apd, 0)
        apd = apd.cuda()
        out = model(apd).squeeze()

        mse = F.mse_loss(out.view(1, -1), u_delta[i].view(1, -1), reduction='mean')            
        total_mse += mse.item()

    total_mse /= ntest
    print (f"The ground truth MSE => MSE(model(a+delta), solver(a+delta)) : {total_mse :.6f}")

# Get the attacked_mse = MSE (model(a+delta), solver(b_j))
def get_attack_mse(model, ap_delta, u, ntest):

    # iterate through ap_delta
    total_mse = 0
    for i in range(ntest):
        apd = ap_delta[i, :, :]
        apd = torch.unsqueeze(apd, 0)
        apd = apd.cuda()
        out = model(apd).squeeze()

        mse = F.mse_loss(out.view(1, -1), u[i].view(1, -1), reduction='mean')            
        total_mse += mse.item()

    total_mse /= ntest
    print (f"The attack MSE => MSE(model(a+delta), solver(b_j)) : {total_mse :.9f}")


def get_apd_pred(model, apd, y_test):
    pred = torch.zeros(y_test.shape)
    test_mse = 0
    N = apd.size(0)
    with torch.no_grad():
        for n in range(N):
            x = apd[n,:,:]
            x = torch.unsqueeze(x, 0)
            x = x.cuda()

            out = model(x).squeeze()
            pred[n] = out
    return pred

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def print_inputs(x_test, a_plus_delta, b_j, eps):
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(x_test[0,:, 0].squeeze(), label='a')
    line2 = ax1.plot(a_plus_delta[0,:], label='a+delta')
    ax2 = ax1.twinx()
    line3 = ax2.plot(b_j[0,:], c='green', label='b_j')

    lines = line1 + line2 + line3 
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
    ax1.set_title(f'input plots with eps={eps}')

def print_outputs(pred, apd_pred, bj_pred, eps):
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(pred[0,:], label='model(a)')
    line2 = ax1.plot(apd_pred[0,:], label='model(a+delta)')
    ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    line3 = ax2.plot(bj_pred[0,:], c='green', label='model(b_j)')
    # line3 = ax2.plot(u_prime[0,:], c='red', label='solver(b_j)') 

    lines = line1 + line2 + line3 
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
    ax1.set_title(f'output plots with eps={eps}')

