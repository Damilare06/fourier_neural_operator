""" Simple attack showing  a ResNet-18 model attack for different epsolons vs accuracy"""

# run with python3.8 single_attack_foolbox.py 
from numpy import imag
from numpy.core.fromnumeric import shape
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

def main() -> None:
    # instantiate the model
    model = models.resnet18(pretrained=True).eval()
    preprocessing =  dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data to test the model, optionally working with eagerpy
    images, labels =  ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    print(type(images), images.shape)
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy: {clean_acc * 100:.1f} %")

    # apply the attack
    attack = LinfPGD()
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
    """

    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    # rAutomatically compute the robust accuracy
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    print(type(robust_accuracy))
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f" Linf norm <= {eps: < 6}: {acc.item() * 100:4.1f} %")

    """

    """
    # to manually repeat the experiment, we use the clipped advs to be certain the experiment falls within 
    print()
    print(" We can also do this check manually;")
    print()
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, advs_, labels)
        print(f" Linf norm <= {eps: < 6}: {acc2 * 100:4.1f} %")
        print("  perturbation sizes:")
        perturb_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("     ", str(perturb_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break
    """


if __name__ == "__main__":
    main()