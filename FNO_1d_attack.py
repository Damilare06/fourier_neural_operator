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

def get_proxy_mse(model, loader, attack, attack_name, xtest, *args):
    total_loss = 0.
    index = 0
    a_plus_delta = torch.zeros_like(xtest)
    delta_arr = torch.zeros_like(xtest)

    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        delta = attack(model, X, y.squeeze(), *args)

        # zero out the loc_delta index
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
    # output x_new = [xtest+delta, loc]


    output =  total_loss / len(loader.dataset)
    print(f"The proxy {attack_name} error => MSE(model(a + delta), model(a)) = : {output :.6f} ")

    return delta_arr[:,:,0].squeeze(), a_plus_delta

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
    print (f"The attack MSE => MSE(model(a+delta), solver(b_j)) : {total_mse :.6f}")


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


def main() -> None:
    #  configurations
    ntest = 100
    sub = 2**3 # subsampling rate
    h = 2**13 // sub # total grid size divided by the subsampling rate
    s = h

    batch_size = 1 # 20
    debug = False 
    evaluate = True # False

    # read data of the shape (number of samples, grid size)
    if evaluate:
        # dataloader = MatReader('data/burgers_N2048_G8192.mat')
        dataloader = MatReader('data/burgers_N2048_G8192_inf_2.mat')
        x_data_full = dataloader.read_field('a')[:,:]
        x_test = x_data_full[-ntest:,::sub]
        y_test = dataloader.read_field('u')[-ntest:,::sub]

        # dataloader2 = MatReader('data/burgers_N100_G1092_B5000_gen.mat')
        # dataloader2 = MatReader('data/burgers_N100_G1092.mat')
        # dataloader2 = MatReader('data/burgers_N100_G1092_e05.mat')
        dataloader2 = MatReader('data/burgers_N100_G1092_e01.mat')
        # dataloader2 = MatReader('data/burgers_N100_G1092_e1.mat')
        b_j = dataloader2.read_field('b_j')[:,:]# or [:,:]
        u_prime = dataloader2.read_field('u')[:ntest,::sub]

        # cat the locations information
        grid = np.linspace(0, 1, s).reshape(1, s, 1)
        grid = torch.tensor(grid, dtype=torch.float)
        x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)

        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)

        # load and initialize the model
        model = torch.load('model/ns_N2048_G8192_burgers').eval()

        # Evaluation - the foolbox accuracy loss is for classification
        pred = torch.zeros(y_test.shape)
        index = 0
        test_mse = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                # print(x.shape)

                out = model(x).squeeze()
                pred[index] = out

                mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')            
                test_mse += mse.item()

                index = index + 1
        test_mse /= ntest
        print(f"The test_mse loss before attack := {test_mse :.9f} ")

        # alpha = step size, epsilon = perturbation range 
        eps = 0.01
        num_iter = 10
        restarts = 10
        alpha = eps/ num_iter

        # proxy_mse := MSE(model(a + delta), model(a))
        delta_out, ap_delta = get_proxy_mse(model, test_loader, mse_attack, "mse_attack", x_test, eps, alpha, num_iter)
        a_plus_delta = ap_delta[:,:,0].squeeze()

        # get_proxy_mse(model, test_loader, mse_linf_attack, "mse_linf_attack", x_test, eps, alpha, num_iter)
        # get_proxy_mse(model, test_loader, mse_linf_rand_attack, "mse_linf_rand_attack", x_test, eps, alpha, num_iter, restarts)

        # POST-PROCESSING
        a_p_delta = x_data_full[-ntest:,:]
        delta = torch.zeros_like(x_data_full[:ntest,:])

        a_p_delta[:,::sub] = a_plus_delta
        delta[:,::sub] = delta_out
        # print("ABJ: ", delta.shape, delta_out.shape)

        scipy.io.savemat('pred/a_p_delta_burger_N2048_G8092_e01.mat', mdict={'a': x_test[:,:,0].cpu().numpy() ,'a_plus_delta': a_p_delta.cpu().numpy(), \
                'delta': delta.cpu().numpy(), 'y_pred': pred.cpu().numpy(), 'delta_sub': delta_out.cpu().numpy(), 'apd_sub': ap_delta.cpu().numpy()})

        ##################################
        # Printing the inputs
        # ##################################
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(x_test[0,:, 0].squeeze(), label='a')
        line2 = ax1.plot(a_plus_delta[0,:], label='a+delta')
        ax2 = ax1.twinx()
        line3 = ax2.plot(b_j[0,:], c='green', label='b_j')

        lines = line1 + line2 + line3 
        ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
        ax1.set_title(f'input plots with eps={eps}')

        ##################################
        # Printing the outputs
        ##################################
        a_plus_delta = torch.cat([a_plus_delta.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)
        b_j_delta = torch.cat([b_j.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)
        # print(a_plus_delta.shape, x_test.shape)
        apd_pred = get_apd_pred(model, a_plus_delta, y_test)
        bj_pred = get_apd_pred(model, b_j_delta, y_test)

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

        plt.show()

    # Ground_truth_mse = MSE(model(a+delta), solver(a+delta))
    # get_ground_truth_mse(model, ap_delta, 'data/burgers_N2048_G8192_gen.mat', 'u', ntest, sub )

    # Get the attacked_mse = MSE (model(a+delta), solver(b_j))
    get_attack_mse(model, ap_delta, u_prime.cuda(), ntest)




if __name__ == "__main__":
    main()