from contextlib import contextmanager

import pandas as pd
from MagmaPandas.MagmaFrames import Olivine

from MagmaPEC.Kd_calculation import calculate_Fe2FeTotal


class variables_container:
    """
    A container for variables
    """

    def __init__(self, **kwargs):

        self._variables = kwargs
        self.sliced = {}

    def reslice(self, samples):

        # make sure samples is list like, so that sliced dataframes remain dataframes (and not series)
        try:
            len(samples)
        except TypeError:
            samples = [samples]

        self.sliced = {
            key: val.loc[samples].copy() for key, val in self._variables.items()
        }

    def fetch(self, var_names):

        return {name: self.sliced[name] for name in var_names}


class FeO_Target:

    def __init__(self, FeO_target, samples: pd.Index):

        self._as_function = False
        self.function = None

        try:
            self._process_arraylike(FeO_target=FeO_target, samples=samples)
        except TypeError:
            if isinstance(FeO_target, (int, float)):
                self._target = pd.Series(FeO_target, index=samples, name="FeO_target")
            elif hasattr(FeO_target, "__call__"):
                self._as_function = True
                self.function = FeO_target

    def target(self, melt_wtpc=None):
        if not self._as_function:
            return self._target.loc[melt_wtpc.index].copy()

        return self.function(melt_wtpc)

    def _process_arraylike(self, FeO_target, samples):
        if len(FeO_target) != len(samples):
            raise ValueError(
                "Number of initial FeO inputs and inclusions does not match"
            )
        if hasattr(FeO_target, "index"):
            if not FeO_target.index.equals(samples):
                raise ValueError("FeO target inputs and inclusion indeces don't match")
            self._target = FeO_target
        else:
            self._target = pd.Series(FeO_target, index=samples, name="FeO_target")


def get_olivine_composition(melt_mol_fractions, Fe3Fe2, Kd):

    Fe2_FeTotal = calculate_Fe2FeTotal(Fe3Fe2=Fe3Fe2)
    melt_FeMg = (melt_mol_fractions["FeO"] * Fe2_FeTotal) / melt_mol_fractions["MgO"]
    # equilibrium olivine composition in oxide mol fractions
    Fo_equilibrium = 1 / (1 + Kd * melt_FeMg)
    olivine = Olivine(
        {
            "MgO": Fo_equilibrium * 2,
            "FeO": (1 - Fo_equilibrium) * 2,
            "SiO2": 1,
        },
        index=melt_mol_fractions.index,
        columns=melt_mol_fractions.columns,
        units="mol fraction",
        datatype="oxide",
        dtype=float,
    )
    olivine = olivine.fillna(
        0.0
    ).normalise()  # why did I replace with 1e-6 instead of 0.0 before?

    return olivine


@contextmanager
def null_progressbar(*args, **kwargs):
    yield lambda *args, **kwargs: None
