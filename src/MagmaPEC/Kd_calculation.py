from typing import Tuple

import pandas as pd
from MagmaPandas.Fe_redox.Fe3Fe2_models import Fe3Fe2_models_dict
from MagmaPandas.fO2.fO2_calculate import calculate_fO2
from MagmaPandas.Kd.Ol_melt.FeMg import Kd_olmelt_FeMg_models_dict

from MagmaPEC import model_configuration


def calculate_Fe2FeTotal(Fe3Fe2, **kwargs):

    return 1 / (1 + Fe3Fe2)


def calculate_observed_Kd(melt_mol_fractions, Fe3Fe2, forsterite):

    melt_mol_fractions = melt_mol_fractions.normalise()
    Fe2_FeTotal = 1 / (1 + Fe3Fe2)
    melt_MgFe = melt_mol_fractions["MgO"] / (melt_mol_fractions["FeO"] * Fe2_FeTotal)
    olivine_MgFe = forsterite / (1 - forsterite)

    return melt_MgFe / olivine_MgFe


def calculate_Kds(
    melt_mol_fractions, forsterite, T_K, P_bar, Fe3Fe2, offset_parameters, **kwargs
) -> Tuple[pd.Series, pd.Series]:
    """
    Model equilibrium Kds and calculate measured Kds
    """

    Kd_model = Kd_olmelt_FeMg_models_dict[model_configuration.Kd_model]

    Kd_observed = calculate_observed_Kd(
        melt_mol_fractions=melt_mol_fractions, Fe3Fe2=Fe3Fe2, forsterite=forsterite
    )

    Kd_equilibrium = Kd_model._calculate_Kd_(
        melt_mol_fractions=melt_mol_fractions,
        T_K=T_K,
        P_bar=P_bar,
        Fe3Fe2=Fe3Fe2,
        forsterite_intial=forsterite,
        offset_parameters=offset_parameters,
        **kwargs,
    )

    if isinstance(Kd_equilibrium, (float, int)):
        return Kd_equilibrium, Kd_observed

    Kd_observed = pd.Series(Kd_observed, index=melt_mol_fractions.index)
    Kd_observed.rename("real", inplace=True)
    Kd_observed.index.name = "name"

    Kd_equilibrium = pd.Series(Kd_equilibrium, index=melt_mol_fractions.index)
    Kd_equilibrium.rename("equilibrium", inplace=True)
    Kd_equilibrium.index.name = "name"

    return Kd_equilibrium, Kd_observed


def _calculate_Kds(melt_mol_fractions, forsterite, P_bar):

    T_K = melt_mol_fractions.temperature(P_bar=P_bar)
    fO2 = calculate_fO2(T_K=T_K, P_bar=P_bar)

    Fe3Fe2_model = Fe3Fe2_models_dict[model_configuration.Fe3Fe2_model]
    Fe3Fe2 = Fe3Fe2_model._calculate_Fe3Fe2_(
        melt_mol_fractions=melt_mol_fractions,
        T_K=T_K,
        fO2=fO2,
        offset_parameters=0.0,
        P_bar=P_bar,
    )
    Kd_equilibrium, Kd_observed = calculate_Kds(
        melt_mol_fractions=melt_mol_fractions,
        forsterite=forsterite,
        T_K=T_K,
        P_bar=P_bar,
        Fe3Fe2=Fe3Fe2,
        offset_parameters=0.0,
        fO2=fO2,
    )

    return Kd_equilibrium, Kd_observed
