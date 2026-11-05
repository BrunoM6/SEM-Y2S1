"""
Two small, optional changes to river.cluster.ODAC (checked against river 0.26.1).

apply()
    River computes the average pairwise distance used in the split heuristic as
    sum(distances) / n_observations. Rodrigues, Gama & Pedroso (2008) use the mean
    of the pairwise rnomc distances in the cluster. With the River version the
    heterogeneity test is almost always true, so near-homogeneous clusters get split
    and later merged back, which shows up as false alerts. This patch uses the mean
    over pairs instead.

apply_grace(min_obs)
    A freshly split child starts with empty statistics, but its parent's diameter was
    frozen on many observations. Testing the child for aggregation after only n_min
    samples compares a noisy estimate with a precise one, and produced split->merge
    flips in healthy runs. A leaf must now have seen `min_obs` observations before it
    can trigger a merge.
"""
from river.cluster import odac as _odac

_orig_calculate = _odac.ODACCluster.calculate_coefficients


def _calculate_coefficients(self, confidence_level: float) -> None:
    _orig_calculate(self, confidence_level)
    if len(self.timeseries_names) > 1:
        d = self._calculate_rnomc_dict()
        self.avg = sum(d.values()) / len(d)


def apply() -> None:
    _odac.ODACCluster.calculate_coefficients = _calculate_coefficients


def apply_grace(min_obs: int) -> None:
    def test_aggregate(self):
        if self.parent is not None and self.d1 is not None and self.n >= min_obs:
            if self.d1 - self.parent.d1 > max(self.parent.e, self.e):
                self.parent._aggregate_this_cluster()
                return True
        return False

    _odac.ODACCluster.test_aggregate = test_aggregate
