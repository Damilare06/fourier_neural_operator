""" 
    Simple attack showing the 2D Darcy model attacks
    Run with "python3.8 single_attack.py"
"""

import gc
import torch
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


def get_proxy_mse(model, loader, attack, attack_name, test_a, *args):
    total_loss = 0.
    index = 0
    delta_arr = torch.zeros_like(test_a)
    a_plus_delta = torch.zeros_like(test_a)

    for X, y in loader:
        X, y = X.cuda(), y.cuda()
        delta = attack(model, X, y.squeeze(), *args)
        yp = model(X + delta)
        loss = F.mse_loss(yp.squeeze(), y.squeeze())
        
        total_loss += loss

    output =  total_loss / len(loader.dataset)
    print(f"{attack_name} error: {output :.6f} ")
    return output

def show_overlap(var1, var2, key1, key2):
    cm = plt.cm.get_cmap('viridis')
    var = [var1.numpy(), var2.numpy()]
    var = np.asarray(var)
    # print(var.shape)
    var = var.reshape(var.shape[1], var.shape[2], var.shape[3], var.shape[4], var.shape[0])
    # print(var.shape)
    key = [key1, key2]

    # Generate the data distribution
    fig, ax = plt.subplots()
    ax.set_title(f"N = {var.shape[0]} of {key[0]} vs {key[1]} on {var.shape[1]} X {var.shape[2]} grid\n")
    var_hist = var.reshape(var.shape[0] * var.shape[1] * var.shape[2] * var.shape[3], var.shape[4])
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
    ################################################################
    # configs
    ################################################################
    TEST_PATH = '/gpfs/u/home/MPFS/MPFSadsj/scratch/fourier_neural_operator/ns_V1e-3_N5000_T50.mat'
    ntest = 20

    modes = 8
    width = 20

    batch_size = 10

    epochs = 500
    learning_rate = 0.001
    scheduler_step = 100
    scheduler_gamma = 0.5

    # print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    sub = 1
    S = 64 // sub
    T_in = 10
    T = 30

    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(TEST_PATH)
    train_buff = reader.read_field('u')[-ntest:,:,:,:]

    test_a = train_buff[-ntest:,::sub,::sub,:T_in]
    test_u = train_buff[-ntest:,::sub,::sub,T_in:T+T_in]


    a_normalizer = UnitGaussianNormalizer(test_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(test_u)
    test_u = y_normalizer.encode(test_u)

    test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])

    # pad locations (x,y,t)
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

    test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                        gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)
    #print("ABJ 0.05", test_a.shape)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda')


    ################################################################
    # training and evaluation
    ################################################################
    model = torch.load('model/navier_test').eval()
    print(count_params(model))
    
    # Evaluation - the foolbox accuracy loss is for classification
    pred = torch.zeros(test_u.shape)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
    index = 0
    test_mse = 0
    batch_size = 1
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).squeeze(axis=4).cpu()
            out = y_normalizer.decode(out)
            pred[index] = out

            mse = F.mse_loss(out.cuda().view(batch_size, -1), y.view(batch_size, -1), reduction='mean')            
            test_mse += mse.item()
            # print(index, test_mse)
            index = index + 1
    test_mse /= ntest
    print(f"test_mse loss before attack: {test_mse :.6f} ")
    # scipy.io.savemat('pred/navier_test.mat', mdict={'pred': pred.cpu().numpy()})

    # show_navier(test_u, 'y_test')
    # show_navier(pred, 'y_pred')
    #print("ABJ: ", test_u.shape, pred.shape)
    # show_overlap(test_u, pred, 'u_test', 'u_pred')
    # plt.show()
    
    # Comment: alpha = step size, epsilon = perturbation range 
    eps = 0.1
    num_iter = 10
    restarts = 10
    alpha = eps/ num_iter

    get_proxy_mse(model, test_loader, mse_attack, "mse_attack", test_a, eps, alpha, num_iter)
    # get_proxy_mse(model, test_loader, mse_linf_rand_attack, "mse_linf_rand_attack", test_a, eps, alpha, num_iter, restarts)
    # get_proxy_mse(model, test_loader, mse_linf_attack, "mse_linf_attack", test_a, eps, alpha, num_iter)


if __name__ == "__main__":
    main()
