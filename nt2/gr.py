from typing import Callable, Tuple
import math


class Metric:
    def __init__(self, name: str):
        self.name = name
        self._h_11 = lambda x: 1
        self._h_22 = lambda x: 1
        self._h_33 = lambda x: 1
        self._h_13 = lambda x: 0

    @property
    def h_11(self) -> Callable[[Tuple[float]], float]:
        return self._h_11

    @property
    def h11(self) -> Callable[[Tuple[float]], float]:
        return lambda x: self.h_33(x) / (
            self.h_11(x) * self.h_33(x) - self.h_13(x) ** 2
        )

    @property
    def h_22(self) -> Callable[[Tuple[float]], float]:
        return self._h_22

    @property
    def h22(self) -> Callable[[Tuple[float]], float]:
        return lambda x: 1 / self.h_22(x)

    @property
    def h_33(self) -> Callable[[Tuple[float]], float]:
        return self._h_33

    @property
    def h33(self) -> Callable[[Tuple[float]], float]:
        return lambda x: self.h_11(x) / (
            self.h_11(x) * self.h_33(x) - self.h_13(x) ** 2
        )

    @property
    def h_13(self) -> Callable[[Tuple[float]], float]:
        return self._h_13

    @property
    def h13(self) -> Callable[[Tuple[float]], float]:
        return lambda x: -self.h_13(x) / (
            self.h_11(x) * self.h_33(x) - self.h_13(x) ** 2
        )

    def v3_Cov2Cntrv(self, u: Tuple[float], x: Tuple[float]) -> Tuple[float]:
        return (
            self.h11(x) * u[0] + self.h13(x) * u[2],
            self.h22(x) * u[1],
            self.h13(x) * u[0] + self.h33(x) * u[2],
        )

    def v3_Cntrv2Cov(self, u: Tuple[float], x: Tuple[float]) -> Tuple[float]:
        return (
            self.h_11(x) * u[0] + self.h_13(x) * u[2],
            self.h_22(x) * u[1],
            self.h_13(x) * u[0] + self.h_33(x) * u[2],
        )

    def v3_Cov2Hat(self, u: Tuple[float], x: Tuple[float]) -> Tuple[float]:
        return (
            math.sqrt(self.h11(x)) * (u[0] - (self.h_13(x) / self.h_33(x)) * u[2]),
            u[1] / math.sqrt(self.h_22(x)),
            u[2] / math.sqrt(self.h_33(x)),
        )

    def v3_Hat2Cov(self, u: Tuple[float], x: Tuple[float]) -> Tuple[float]:
        return (
            u[0] / math.sqrt(self.h11(x))
            + (self.h_13(x) / math.sqrt(self.h_33(x))) * u[2],
            u[1] * math.sqrt(self.h_22(x)),
            u[2] * math.sqrt(self.h_33(x)),
        )

    def v3_Cntrv2Hat(self, u: Tuple[float], x: Tuple[float]) -> Tuple[float]:
        return self.v3_Cov2Hat(self.v3_Cntrv2Cov(u, x), x)

    def v3_Hat2Cntrv(self, u: Tuple[float], x: Tuple[float]) -> Tuple[float]:
        return self.v3_Cov2Cntrv(self.v3_Hat2Cov(u, x), x)


class Minkowski(Metric):
    def __init__(self):
        super().__init__("Minkowski")


class KerrSchild(Metric):
    def __init__(self, a: float):
        super().__init__("Kerr-Schild")
        self._a = a
        self.alpha = lambda x: 1 / math.sqrt(
            1 + 2 * x[0] / (x[0] + self.a**2 * math.cos(x[1]) ** 2)
        )
        self.beta1 = lambda x: (
            2
            * x[0]
            / (x[0] ** 2 + self.a**2 * math.cos(x[1]) ** 2)
            / (1 + 2 * x[0] / (x[0] ** 2 + self.a**2 * math.cos(x[1]) ** 2))
        )

        self._h_11 = lambda x: (
            1 + 2 * x[0] / (x[0] ** 2 + self.a**2 * math.cos(x[1]) ** 2)
        )
        self._h_22 = lambda x: (x[0] ** 2 + self.a**2 * math.cos(x[1]) ** 2)
        self._h_33 = lambda x: (
            (
                (x[0] ** 2 + self.a**2) * (x[0] ** 2 + self.a**2)
                - self.a**2
                * (x[0] ** 2 - 2 * x[0] + self.a**2)
                * math.sin(x[1]) ** 2
            )
            * math.sin(x[1]) ** 2
            / (x[0] ** 2 + self.a**2 * math.cos(x[1]) ** 2)
        )
        self._h_13 = (
            lambda x: self.a
            * math.sin(x[1])
            * (1 + 2 * x[0] / (x[0] ** 2 + self.a**2 * math.cos(x[1]) ** 2))
        )

    @property
    def a(self) -> float:
        return self._a


class Schwarzschild(KerrSchild):
    def __init__(self):
        super().__init__(0)
        self.name = "Schwarzschild"


class FourVector:
    def __init__(self, u: Tuple[float], x: Tuple[float], metric: Metric):
        assert len(u) == 4, "FourVector must be 4-dimensional"
        self._u = u
        self._x = x
        self._metric = metric

    @property
    def metric(self) -> Metric:
        return self._metric

    @property
    def x(self) -> Tuple[float]:
        return self._x

    @property
    def u(self) -> Tuple[float]:
        return self._u

    @property
    def spatial(self) -> Tuple[float]:
        return self.u[1:]

    @property
    def cntrv(self) -> Tuple[float]:
        return self.u[1:]

    @property
    def cov(self) -> Tuple[float]:
        return self.metric.v3_Cntrv2Cov(self.cntrv, self.x)

    @property
    def norm(self) -> float:
        return math.sqrt(sum([u1 * u_1 for u1, u_1 in zip(self.cntrv, self.cov)]))

    def __repr__(self) -> str:
        return f"({self.u[0]}; {self.u[1]}, {self.u[2]}, {self.u[3]}) | {self.x}, {self.metric.name}"


class Photon(FourVector):
    def __init__(self, u: Tuple[float], x: Tuple[float], metric: Metric):
        assert len(u) == 3, "The Photon class must be given a 3-dimensional vector"
        super().__init__((0, *u), x, metric)
        self._u = (self.norm, *self.spatial)
