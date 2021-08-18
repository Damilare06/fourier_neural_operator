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
from auxilliary_1D import *

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
        dataloader2 = MatReader('data/burgers_N100_G1092_e05_l2.mat')
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
        eps = 0.05
        num_iter = 10
        restarts = 10
        alpha = eps/ num_iter

        # proxy_mse := MSE(model(a + delta), model(a))
        # delta_out, ap_delta = pgd_linf(model, test_loader, mse_attack, "mse_attack", x_test, eps, alpha, num_iter)
        delta_out, ap_delta = pgd_l2(model, x_test.cuda(), y_test.cuda(), eps, alpha, num_iter)
        # print(delta_out.shape, delta_out.get_device())
        a_plus_delta = ap_delta[:,:,0].squeeze()

        # delta_out, ap_delta = pgd_linf(model, test_loader, mse_linf_attack, "mse_linf_attack", x_test, eps, alpha, num_iter)
        # delta_out, ap_delta = pgd_linf(model, test_loader, mse_linf_rand_attack, "mse_linf_rand_attack", x_test, eps, alpha, num_iter, restarts)

        # POST-PROCESSING
        a_p_delta = x_data_full[-ntest:,:]
        delta = torch.zeros_like(x_data_full[:ntest,:])

        a_p_delta[:,::sub] = a_plus_delta
        delta[:,::sub] = delta_out

        # Printing the inputs & outputs
        print_inputs(x_test, a_plus_delta, b_j, eps)

        a_plus_delta = torch.cat([a_plus_delta.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)
        b_j_delta = torch.cat([b_j.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)
        apd_pred = get_apd_pred(model, a_plus_delta, y_test)
        bj_pred = get_apd_pred(model, b_j_delta, y_test)
        
        print_outputs(pred, apd_pred, bj_pred, eps)
        # plt.show()

        scipy.io.savemat('pred/a_p_delta_burger_N2048_G8092_e05_l2.mat', mdict={'a': x_test[:,:,0].cpu().numpy() ,'a_plus_delta': a_p_delta.cpu().numpy(), \
                'delta': delta.cpu().numpy(), 'y_pred': pred.cpu().numpy(), 'delta_sub': delta_out.cpu().numpy(), 'apd_sub': ap_delta.cpu().numpy(), \
                    'apd_pred': apd_pred.cpu().numpy()})

    # Ground_truth_mse = MSE(model(a+delta), solver(a+delta))
    # get_ground_truth_mse(model, ap_delta, 'data/burgers_N2048_G8192_gen.mat', 'u', ntest, sub )

    # Get the attacked_mse = MSE (model(a+delta), solver(b_j))
    get_attack_mse(model, ap_delta, u_prime.cuda(), ntest)




if __name__ == "__main__":
    main()