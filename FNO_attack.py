# its inherits from BaseGradientDescent
from foolbox.distances import linf
from foolbox.attacks.gradient_descent_base import LinfBaseGradientDescent, BaseGradientDescent

class FNO_LinfPGD(BaseGradientDescent):
    distance = linf

# class LinfBaseGradientDescent(BaseGradientDescent):
#     distance = linf

#     def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
#         return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

#     def normalize(
#         self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
#     ) -> ep.Tensor:
#         return gradients.sign()

#     def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
#         return x0 + ep.clip(x - x0, -epsilon, epsilon)