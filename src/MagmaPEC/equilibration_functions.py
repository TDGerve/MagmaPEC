import math

import numpy as np
from MagmaPandas.Fe_redox.Fe3Fe2_models import Fe3Fe2_models_dict
from MagmaPandas.fO2.fO2_calculate import calculate_fO2
from scipy.optimize import root_scalar

from MagmaPEC import model_configuration
from MagmaPEC.Kd_calculation import calculate_Kds


def _root_temperature(
    olivine_amount,
    melt_mol_fractions,
    olivine_mol_fractions,
    T_K,
    P_bar,
    temperature_offset,
):

    melt_x_new = melt_mol_fractions.add(olivine_mol_fractions.mul(olivine_amount))
    melt_x_new = melt_x_new.normalise()
    temperature_new = melt_x_new.temperature(
        P_bar=P_bar, offset=temperature_offset, warn=True
    )

    return T_K - temperature_new


def _root_Kd(
    exchange_amount,
    melt_mol_fractions,
    exchange_vector,
    forsterite,
    P_bar,
    Fe3Fe2_offset_parameters,
    Kd_offset_parameters,
    kwargs,
):

    melt_x_new = melt_mol_fractions.add(exchange_vector.mul(exchange_amount))
    melt_x_new = melt_x_new.normalise()

    T_K = melt_x_new.temperature(P_bar=P_bar)
    fO2 = calculate_fO2(T_K=T_K, P_bar=P_bar)

    Fe3Fe2_model = Fe3Fe2_models_dict[model_configuration.Fe3Fe2_model]
    Fe3Fe2 = Fe3Fe2_model._calculate_Fe3Fe2_(
        melt_mol_fractions=melt_x_new,
        T_K=T_K,
        P_bar=P_bar,
        fO2=fO2,
        offset_parameters=Fe3Fe2_offset_parameters,
    )

    Kd_equilibrium, Kd_real = calculate_Kds(
        melt_mol_fractions=melt_x_new,
        forsterite=forsterite,
        T_K=T_K,
        P_bar=P_bar,
        Fe3Fe2=Fe3Fe2,
        offset_parameters=Kd_offset_parameters,
        fO2=fO2,
        **kwargs,
    )

    return Kd_equilibrium - Kd_real


def isothermal_equilibration(
    melt_mol_fractions, olivine_moles, T_K, P_bar, temperature_offset
):
    """
    Isothermal equilibration via olivine melting or crystallisation.
    """

    min_MgO = -(melt_mol_fractions["MgO"] / olivine_moles["MgO"])
    min_FeO = -melt_mol_fractions["FeO"] / olivine_moles["FeO"]
    min_val = max(min_MgO, min_FeO) + 1e-3

    try:
        olivine_amount = root_scalar(
            _root_temperature,
            args=(
                melt_mol_fractions,
                olivine_moles,
                T_K,
                P_bar,
                temperature_offset,
            ),
            # x0=0,
            # x1=sign * 0.01,
            bracket=[min_val, 10],
        ).root
        equilibration = True if not math.isnan(olivine_amount) else False

    except (ValueError, TypeError):
        olivine_amount = np.nan
        equilibration = False

    return olivine_amount, equilibration


def diffusive_equilibration(
    melt_mol_fractions, forsterite, FeMg_vector, P_bar, offset_parameters
):
    """
    Equilibration via diffusive exchange of Fe and Mg
    """

    # set limits to ensure melt FeO or MgO does not go to 0.
    Fe_sign = math.copysign(1, FeMg_vector["FeO"])
    Mg_sign = math.copysign(1, FeMg_vector["MgO"])

    Mg_limit = melt_mol_fractions["MgO"] * Fe_sign  # don't let MgO or FeO go to 0
    Fe_limit = melt_mol_fractions["FeO"] * Mg_sign

    min_val, max_val = min(Mg_limit, Fe_limit) + 1e-3, max(Mg_limit, Fe_limit) - 1e-3
    try:
        exchange_amount = root_scalar(
            _root_Kd,
            args=(
                melt_mol_fractions,
                FeMg_vector,
                forsterite,
                P_bar,
                offset_parameters["Fe3Fe2"],
                offset_parameters["Kd"],
                {},
            ),
            # x0=0,
            # x1=max_val / 10,
            bracket=[min_val, max_val],
        ).root

        equilibration = True if not math.isnan(exchange_amount) else False

    except (ValueError, FloatingPointError):
        exchange_amount = 0.0
        equilibration = False

    return exchange_amount, equilibration
