"""
L2AdditiveUniformNoiseAttackRegr - similar to the noise addition for classification
    It depends on the L2Mixin, GaussianMixin and BaseAdditiveNoiseAttack classes

    the suffix ...Regr is used to denote routines to be modified for Regr
"""
# its inherits from BaseGradientDescent
from typing import Callable, TypeVar, Any, Union, Optional, Sequence, List, Tuple, Dict

from foolbox.distances import Distance, l2
from foolbox.attacks.gradient_descent_base import LinfBaseGradientDescent, BaseGradientDescent
from foolbox.attacks.base import AttackWithDistance, Criterion
from foolbox.attacks.additive_noise import L2Mixin,UniformMixin
from foolbox.models.base import Model
from typing_extensions import final, overload
from abc import ABC, abstractmethod
from collections.abc import Iterable
import eagerpy as ep

T = TypeVar("T")

# class Attack(ABC):
class AttackRegr(ABC):
    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    @abstractmethod  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:
        # in principle, the type of criterion is Union[Criterion, T]
        # but we want to give subclasses the option to specify the supported
        # criteria explicitly (i.e. specifying a stricter type constraint)
        ...

    @abstractmethod
    def repeat(self, times: int) -> "Attack":
        ...

    def __repr__(self) -> str:
        args = ", ".join(f"{k.strip('_')}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({args})"


# class AttackWithDistance(Attack):
class AttackWithDistanceRegr(AttackRegr):
    @property
    @abstractmethod
    def distance(self) -> Distance:
        ...

    def repeat(self, times: int) -> AttackRegr:
        return Repeated(self, times)

# class Repeated(AttackWithDistance):
class Repeated(AttackWithDistanceRegr):
    """Repeats the wrapped attack and returns the best result"""

    def __init__(self, attack: AttackWithDistance, times: int):
        if times < 1:
            raise ValueError(f"expected times >= 1, got {times}")  # pragma: no cover

        self.attack = attack
        self.times = times

    @property
    def distance(self) -> Distance:
        return self.attack.distance

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    def __call__(  # noqa: F811
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        for i in range(self.times):
            # run the attack
            xps, xpcs, success = self.attack(
                model, x, criterion, epsilons=epsilons, **kwargs
            )
            assert len(xps) == K
            assert len(xpcs) == K
            for xp in xps:
                assert xp.shape == x.shape
            for xpc in xpcs:
                assert xpc.shape == x.shape
            assert success.shape == (K, N)

            if i == 0:
                best_xps = xps
                best_xpcs = xpcs
                best_success = success
                continue

            # TODO: test if stacking the list to a single tensor and
            # getting rid of the loop is faster

            for k, epsilon in enumerate(epsilons):
                first = best_success[k].logical_not()
                assert first.shape == (N,)
                if epsilon is None:
                    # if epsilon is None, we need the minimum

                    # TODO: maybe cache some of these distances
                    # and then remove the else part
                    closer = self.distance(x, xps[k]) < self.distance(x, best_xps[k])
                    assert closer.shape == (N,)
                    new_best = ep.logical_and(success[k], ep.logical_or(closer, first))
                else:
                    # for concrete epsilon, we just need a successful one
                    new_best = ep.logical_and(success[k], first)
                new_best = atleast_kd(new_best, x.ndim)
                best_xps[k] = ep.where(new_best, xps[k], best_xps[k])
                best_xpcs[k] = ep.where(new_best, xpcs[k], best_xpcs[k])

            best_success = ep.logical_or(success, best_success)

        best_xps_ = [restore_type(xp) for xp in best_xps]
        best_xpcs_ = [restore_type(xpc) for xpc in best_xpcs]
        if was_iterable:
            return best_xps_, best_xpcs_, restore_type(best_success)
        else:
            assert len(best_xps_) == 1
            assert len(best_xpcs_) == 1
            return (
                best_xps_[0],
                best_xpcs_[0],
                restore_type(best_success.squeeze(axis=0)),
            )

    def repeat(self, times: int) -> "Repeated":
        return Repeated(self.attack, self.times * times)





# class FixedEpsilonAttackRegr(AttackWithDistance):
class FixedEpsilonAttackRegr(AttackWithDistanceRegr):
    """Fixed-epsilon attacks try to find adversarials whose perturbation sizes
    are limited by a fixed epsilon"""

    @abstractmethod
    def run(
        self, model: Model, inputs: T, criterion: Any, *, epsilon: float, **kwargs: Any
    ) -> T:
        """Runs the attack and returns perturbed inputs.
        The size of the perturbations should be at most epsilon, but this
        is not guaranteed and the caller should verify this or clip the result.
        """
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    @final  # noqa: F811
    def __call__(  # type: ignore
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:

        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        # None means: just minimize, no early stopping, no limit on the perturbation size
        if any(eps is None for eps in epsilons):
            # TODO: implement a binary search
            raise NotImplementedError(
                "FixedEpsilonAttack subclasses do not yet support None in epsilons"
            )
        real_epsilons = [eps for eps in epsilons if eps is not None]
        del epsilons

        xps = []
        xpcs = []
        success = []
        for epsilon in real_epsilons:
            xp = self.run(model, x, criterion, epsilon=epsilon, **kwargs)

            # clip to epsilon because we don't really know what the attack returns;
            # alternatively, we could check if the perturbation is at most epsilon,
            # but then we would need to handle numerical violations;
            xpc = self.distance.clip_perturbation(x, xp, epsilon)
            is_adv = is_adversarial(xpc)

            xps.append(xp)
            xpcs.append(xpc)
            success.append(is_adv)

        success_ = ep.stack(success)
        assert success_.shape == (K, N)

        xps_ = [restore_type(xp) for xp in xps]
        xpcs_ = [restore_type(xpc) for xpc in xpcs]

        if was_iterable:
            return xps_, xpcs_, restore_type(success_)
        else:
            assert len(xps_) == 1
            assert len(xpcs_) == 1
            return xps_[0], xpcs_[0], restore_type(success_.squeeze(axis=0))


class BaseAdditiveNoiseAttackRegr(FixedEpsilonAttackRegr, ABC):
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion, kwargs

        min_, max_ = model.bounds
        p = self.sample_noise(x)
        epsilons = self.get_epsilons(x, p, epsilon, min_=min_, max_=max_)
        x = x + epsilons * p
        x = x.clip(min_, max_)

        return restore_type(x)

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        raise NotImplementedError


class L2Mixin:
    distance = l2

    def get_epsilons(
        self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float
    ) -> ep.Tensor:
        norms = flatten(p).norms.l2(axis=-1)
        return epsilon / atleast_kd(norms, p.ndim)


class L2AdditiveUniformNoiseAttackRegr(L2Mixin, UniformMixin, BaseAdditiveNoiseAttackRegr):
    """Samples uniform noise with a fixed L2 size."""

    pass



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