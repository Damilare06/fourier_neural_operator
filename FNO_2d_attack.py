""" 
    Simple attack showing the 2D Darcy model attacks
    Run with "python3.8 single_attack.py"
		on AiMOS, module load gcc/9.3.0/1 
"""

import torch
import torchvision.models as models
import torch.nn.functional as F
from fourier_attack_module import * 
from utilities3 import *
from auxilliary_2D import *
import sys

# def mse_attack(model, X, y, epsilon=0.1, alpha=1e-5, num_iter=1):
#     """ 
#         Construct PGD-like MSE attacks on X with minimal default values"
#     """
#     delta = torch.zeros_like(X, requires_grad=True)
# 
#     for t in range(num_iter):
#         loss = F.mse_loss(model(X + delta).squeeze(), y)
#         loss.backward()
#         delta.data = (delta + X.shape[0] * alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#         delta.grad.zero_()
#     return delta.detach()
# 
# def mse_linf_attack(model, X, y, epsilon=0.02, alpha=1e-3, num_iter=40):
#     """ 
#         Construct PGD-like MSE attacks on X"
#     """
#     delta = torch.zeros_like(X, requires_grad=True)
# 
#     for t in range(num_iter):
#         loss = F.mse_loss(model(X + delta).squeeze(), y)
#         loss.backward()
#         delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#         delta.grad.zero_()
#     return delta.detach()
# 
# 
# def mse_linf_rand_attack(model, X, y, epsilon, alpha, num_iter, restarts):
#     max_loss = torch.zeros(y.shape[0]).cuda()
#     max_delta = torch.zeros_like(X.squeeze())
#     """ 
#         Construct PGD-like MSE attacks on X"
#     """
#     for i in range(restarts):
#         delta = torch.rand_like(X.squeeze(), requires_grad=True)
#         delta.data = delta.data * 2 * epsilon - epsilon
# 
#         for t in range(num_iter):
#             loss = F.mse_loss(model(X + delta).squeeze(), y)
#             loss.backward()
#             delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#             delta.grad.zero_()
# 
#         all_loss = F.mse_loss(model(X + delta).squeeze(), y, reduction='none')            
#         max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
#         max_loss = torch.max(max_loss, all_loss)
# 
#     return max_delta
# 
# 
# def get_proxy_mse(model, loader, attack, attack_name, x_test, *args):
#     index = 0
#     total_loss = 0.
#     delta_arr = torch.zeros_like(x_test)
#     a_plus_delta = torch.zeros_like(x_test)
# 
#     for X, y in loader:
#         X, y = X.cuda(), y.cuda()
#         delta = attack(model, X, y.squeeze(), *args)
# 
#         # zero out the loc_delta index
#         if 'rand' in attack_name:
#             delta[:,:,1] = 0
#             delta[:,:,2] = 0
#         else:
#             delta[:,:,:,1] = 0
#             delta[:,:,:,2] = 0
# 
#         yp = model(X + delta)
#         loss = F.mse_loss(yp.squeeze(), y.squeeze())
#         
#         total_loss += loss
#         a_plus_delta[index,:,:,:] = X + delta
#         delta_arr[index,:,:,:] = delta
# 
#         index += 1
# 
#     output =  total_loss / len(loader.dataset)
#     print(f"The proxy {attack_name} error => MSE(model(a + delta), model(a)) = : {output :.6f} ")
# 
#     return delta_arr[:,:,:,0].squeeze(), a_plus_delta
# 
# def show_overlap(var1, var2, key1, key2):
#     cm = plt.cm.get_cmap('viridis')
#     var = [var1.numpy(), var2.numpy()]
#     var = np.asarray(var)
#     var = var.reshape(var.shape[1], var.shape[2], var.shape[3], var.shape[0])
#     key = [key1, key2]
# 
#     # Generate the data distribution
#     fig, ax = plt.subplots()
#     ax.set_title(f"N = {var.shape[0]} of {key[0]} vs {key[1]} on {var.shape[1]} grid\n")
#     var_hist = var.reshape(var.shape[0] * var.shape[1] * var.shape[2], var.shape[3])
#     ax.hist(var_hist, density=True, stacked=False, bins='auto', label=key)
#     ax.set_xlabel(f"{key} distribution")
#     ax.legend()
# 
# def show_burgers(var, key):
#     cm = plt.cm.get_cmap('viridis')
#     var = var.numpy()
# 
#     # Generate the data distribution
#     fig, ax = plt.subplots()
#     ax.set_title(f"N = {var.shape[0]} of {key} on {var.shape[1]} grid\n")
#     var_hist = var.flatten()
#     n, bins, patches = ax.hist(var_hist, density=True, stacked=True, bins='auto', color="green")
#     ax.set_xlabel(f"{key} distribution")
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
# 
#     # scale values to interval [0,1]
#     col = bin_centers - min(bin_centers)
#     col /= max(col)
# 
#     for c, p in zip(col, patches):
#         plt.setp(p, 'facecolor', cm(c))    
#     plt.show()
# 
# # Get the attacked_mse = MSE (model(a+delta), solver(b_j))
# def get_attack_mse(model, ap_delta, u, ntest):
#     total_mse = 0
#     for i in range(ntest):
#         apd = np.squeeze(ap_delta[i, :, :, :])
#         apd = torch.unsqueeze(apd, 0)
# #        apd = np.squeeze(ap_delta[i, :, :, 0])
# #        print(apd.shape)
# #        print(u.shape)
# #        apd = torch.unsqueeze(apd, 0)
# #        print(apd.shape)
#         apd = apd.cuda()
#         out = model(apd).squeeze()
#         mse = F.mse_loss(out, u[i].squeeze())
#         # print('ABJ mse: ', mse, out.shape)
#        #  total_mse += mse.item()
#         total_mse += mse
#     
#     total_mse /= ntest
#     print (f"The attack MSE => MSE(model(a+delta), solver(a+delta)) : {total_mse :.6f}")
#  
# 
# def get_apd_pred(model, apd, y_test):
#     pred = torch.zeros(y_test.shape)
#     test_mse = 0
#     N = apd.size(0)
#     with torch.no_grad():
#         for n in range(N):
#             x = apd[n,:,:, :]
#             x = torch.unsqueeze(x, 0)
#             x = x.cuda()
# 
#             out = model(x).squeeze()
#             pred[n] = out
#     return pred

def main() -> None:
    ################################################################
    # configs
    ################################################################
    # TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'
    TEST_PATH = '/gpfs/u/home/MPFS/MPFSadsj/scratch/fourier_neural_operator/darcy_r256_N1000_test.mat'

    ntest = 100
    batch_size = 20
    learning_rate = 0.001

    epochs = 500
    step_size = 100
    gamma = 0.5

    modes = 12
    width = 32
	
    r = 5
    g = 256
    h = int(((g - 1)/r) + 1)
    s = h

    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

    # load a+delta data
    # dataloader2 = MatReader('data/burgers_N100_G1092_B1000_gen.mat')
    # dataloader2 = MatReader('/gpfs/u/home/MPFS/MPFSadsj/scratch/fourier_neural_operator/darcy_r256_N5000.mat')
    # dataloader2 = MatReader('/gpfs/u/home/MPFS/MPFSadsj/scratch/fourier_neural_operator/darcy_r256_N100_2.mat')
    # dataloader2 = MatReader('/gpfs/u/home/MPFS/MPFSadsj/scratch/fourier_neural_operator/darcy_r256_N100_clipped.mat')
    # dataloader2 = MatReader('/gpfs/u/home/MPFS/MPFSadsj/scratch/fourier_neural_operator/darcy_r256_N100_clipped_2.mat')
    # dataloader2 = MatReader('/gpfs/u/home/MPFS/MPFSadsj/scratch/fourier_neural_operator/darcy_r256_N100_e1.mat')
    dataloader2 = MatReader('/gpfs/u/home/MPFS/MPFSadsj/scratch/matlab/Burgers eqn/data_generation/darcy/darcy_r256_N100_e05_l2.mat')
    x_prime = dataloader2.read_field('coeff')[:ntest, ::r, ::r]
    y_prime = dataloader2.read_field('sol')[:ntest, ::r, ::r]


    x_normalizer = UnitGaussianNormalizer(x_test)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_test)
    y_test = y_normalizer.encode(y_test)

    x_normalizer = UnitGaussianNormalizer(x_prime)
    x_prime = x_normalizer.encode(x_prime)
    y_normalizer = UnitGaussianNormalizer(y_prime)
    y_prime = y_normalizer.encode(y_prime)

    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s,s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)
    # print("ABJ: 0.0", x_test.shape)


    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)


    ################################################################
    # training and evaluation
    ################################################################
    model = torch.load('model/ns_fourier_darcy_256').eval()
    # print(count_params(model))
    
    # Evaluation - the foolbox accuracy loss is for classification
    pred = torch.zeros(y_test.shape)
    pred = torch.zeros(y_test.shape)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    index = 0
    test_mse = 0
    batch_size = 1
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            # print(f'ABJ x: {x.shape}')
            out = model(x)
            pred[index] = out.squeeze()

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')            
            test_mse += mse.item()
            # print(index, test_mse)
            # print(f"ABJ x: {x.shape}")
            index = index + 1
    test_mse /= ntest
    print(f"test_mse loss before attack: {test_mse :.6f} ")

    show_burgers(y_test, 'y_test')
    # show_burgers(pred, 'y_pred')
    # show_overlap(y_test, pred, 'u_test', 'u_pred')
    plt.show()
    
    # Comment: alpha = step size, epsilon = perturbation range 
    eps = 0.05
    num_iter = 10
    restarts = 10
    alpha = eps/ num_iter

    delta_norm, ap_delta = get_proxy_mse(model, test_loader, mse_attack, "mse_attack", x_test, eps, alpha, num_iter)

    #TODO: Translate delta and ap_delta to the non-normalized space
    # print("ABJ: ",delta_norm.shape, ap_delta.shape)
    delta_non_norm = x_normalizer.decode(delta_norm)
    apd_non_norm = x_normalizer.decode(ap_delta[:,:,:,0].squeeze())
    #apd_normalizer = UnitGaussianNormalizer(ap_delta)
    #apd_non_norm = x_normalizer.decode(ap_delta[:,:,:,0].squeeze())

    # delta_norm, ap_delta = get_proxy_mse(model, test_loader, mse_linf_attack, "mse_linf_attack", x_test, eps, alpha, num_iter)
    # delta_norm, ap_delta = get_proxy_mse(model, test_loader, mse_linf_rand_attack, "mse_linf_rand_attack", x_test, eps, alpha, num_iter, restarts)
    # delta_norm, ap_delta = pgd_linf(model, test_loader, mse_attack, "mse_attack", x_test, eps, alpha, num_iter)
    delta_norm, ap_delta = pgd_l2(model, x_test.cuda(), y_test.cuda(), eps, alpha, num_iter)

    a_plus_delta = ap_delta# [:,:,:,0].squeeze()

    a_p_delta = reader.read_field('coeff')[:ntest,:,:]
    delta = torch.zeros_like(a_p_delta)

    a_p_delta[:,::r,::r][:,:s,:s] = a_plus_delta[:,:,:,0]
    delta[:,::r,::r] = delta_non_norm
    x_test_send = reader.read_field('coeff')[:ntest,:,:]
    xt_normalizer = UnitGaussianNormalizer(x_test_send)
    x_test_send = xt_normalizer.encode(x_test_send)

    # scipy.io.savemat('pred/a_p_delta_darcy_r256_N100.mat', mdict={'a': x_test[:,:,:,0].cpu().numpy() ,'a_plus_delta': a_p_delta.cpu().numpy(), \
    # 'delta': delta.cpu().numpy(), 'y_pred': pred.cpu().numpy(), 'delta_sub': delta_norm.cpu().numpy(), 'apd_sub':ap_delta.cpu().numpy(), \
    # 'apd_nn':apd_non_norm[:,:,:,0].cpu().numpy(), 'a_orig': x_test_send.cpu().numpy()})
    # scipy.io.savemat('pred/a_p_delta_darcy_r256_N100_e1.mat', mdict={'a': x_test[:,:,:,0].cpu().numpy(),  'delta': delta.cpu().numpy(), \
    # 'delta_sub': delta_norm.cpu().numpy(), 'apd_nn':apd_non_norm.cpu().numpy(), 'a_orig': x_test_send.cpu().numpy()})
    
    # Preprocessing the x_prime
    # 1) normalize it then
    # 2) concat the locations to apd


    x_prime = torch.cat([x_prime.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)
    get_attack_mse(model, ap_delta, y_prime.cuda(), ntest)
    prime_pred = get_apd_pred(model, x_prime, y_test.cuda())
    scipy.io.savemat('pred/apd_darcy_r256_N100_e05_l2.mat', mdict={'a': x_test[:,:,:,0].cpu().numpy(), 'x_prime': x_prime[:,:,:,0].cpu().numpy() , \
    'delta_norm_sub': delta_norm.cpu().numpy(), 'y_pred': pred.cpu().numpy(), 'prime_pred':prime_pred.cpu().numpy(), 'apd':ap_delta.cpu().numpy(), \
    'apd_nn':apd_non_norm.cpu().numpy(), 'a_orig': x_test_send.cpu().numpy(), 'delta': delta.cpu().numpy() })

if __name__ == "__main__":
    main()
