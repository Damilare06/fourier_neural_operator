""" Simple attack showing the Burgers 1d model attack for different epsolons vs accuracy"""

# run with python3.8 single_attack_foolbox.py 
from numpy import exp
import torch
import torchvision.models as models
from foolbox import PyTorchModel, accuracy, samples
from foolbox.distances import linf
import foolbox.attacks as fa
import eagerpy as ep
import torch.nn.functional as F
from fourier_1d_module import * 
from utilities3 import *

def mse_attack(model, X, y, epsilon=0.2, alpha=1e-2, num_iter=40):
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


def main() -> None:

    #  configurations
    ntest = 100

    sub = 2**3 # subsampling rate
    h = 2**13 // sub # total grid size divided by the subsampling rate
    s = h

    batch_size = 1 # 20
    debug = True

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
    preprocessing =  dict(mean=[0.485], std=[0.229], axis=-1)
    fmodel = PyTorchModel(model, bounds=(-1e10, 1e10), preprocessing=preprocessing)

    # Evaluation - the foolbox accuracy loss is for classification
    myloss = LpLoss(size_average=False)

    test_l2 = 0.0
    pred = torch.zeros(y_test.shape)
    index = 0
    test_mse = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            pred[index] = out.squeeze()

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')            
            test_mse += mse.item()

            index = index + 1
    test_l2 /= ntest
    test_mse /= ntest
    # print(f"test_l2 loss: {test_l2 * 100 :.4f} %")
    print(f"test_mse loss: {test_mse * 100 :.4f} %")

    # Apply the attack
    # attacks = [
    #     fa.FGSM(),    # Crossentropy Error
    #     fa.LinfPGD(),   # Crossentropy Error
    #     fa.LinfBasicIterativeAttack(),    # Crossentropy Error
    #     fa.LinfAdditiveUniformNoiseAttack(),
    #     fa.LinfDeepFoolAttack(),    # ValueError: expected the model output to have atleast 2 classes, got 1
    # ]
    attack = fa.LinfAdditiveUniformNoiseAttack()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    print("epsilons")
    print(epsilons)
    print("")

    attack_success = np.zeros((1, len(epsilons), len(x_test)), dtype=bool)
    x_test, y_test = x_test.cuda(), y_test.cuda()
    x_test, y_test = ep.astensors(x_test, y_test)

    # compute the regression error for the fmodel
    out_test = fmodel(x_test).squeeze()

    clean_err = F.mse_loss(y_test.raw, out_test.raw)
    print(f"mse error: {clean_err :.4f} ")
    # clean_acc = torch.exp(-clean_err)
    # print(f"clean_acc: {clean_acc :.4f} ")
    
    # _, clipped, _ = attack(fmodel, x_test, y_test, epsilons=epsilons)


    # TODO: Perturb the input & infer
    # Get the last 100 test set, add gaussian noise to it and draw some inference
    # Does this make any sense? adding this noise to the input? Wouldnt we have to train on the MSE loss

    # TODO: Perturb the input (delta must < epsilon) and Train the MSE Loss on it, clip afterward
    if debug:   print(type(y_test.raw), (y_test.raw).shape)
    delta = mse_attack(model, x_test.raw, y_test.raw, epsilon=0.2, alpha=1e-2)
    if debug:   print(type(delta), (delta).shape)

    y_pred = model(x_test.raw + delta).squeeze()
    pert_err = F.mse_loss(y_test.raw, y_pred)
    print(f"perturbed mse error: {pert_err :.4f} ")

    # # Automatically compute the robust accuracy
    # robust_accuracy = 1 - success.float32().mean(axis=-1)
    # print(type(robust_accuracy))
    # print("robust accuracy for perturbations with")
    # for eps, acc in zip(epsilons, robust_accuracy):
    #     print(f" Linf norm <= {eps: < 6}: {acc.item() * 100:4.1f} %")



if __name__ == "__main__":
    main()