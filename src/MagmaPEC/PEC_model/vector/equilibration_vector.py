import math
import warnings as w
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Progress bar for lengthy calculations
from alive_progress import alive_bar, config_handler

config_handler.set_global(
    title_length=17,
    manual=True,
    theme="smooth",
    spinner=None,
    stats=False,
    length=30,
    force_tty=True,
)

from MagmaPandas.Fe_redox.Fe3Fe2_models import Fe3Fe2_models_dict
from MagmaPandas.fO2.fO2_calculate import calculate_fO2

from MagmaPEC import model_configuration
from MagmaPEC.equilibration_functions import isothermal_equilibration
from MagmaPEC.Kd_calculation import calculate_Kds
from MagmaPEC.PEC_configuration import PEC_configuration
from MagmaPEC.tools import (
    get_olivine_composition,
    null_progressbar,
    variables_container,
)


class equilibration:
    """ """

    _minimum_stepsize = 1e-6

    def __init__(self, inclusions, olivines, P_bar, offset_parameters):

        self.inclusions = inclusions.copy()
        self._inclusions_uncorrected = self.inclusions.copy()
        self.P_bar = P_bar.copy()
        self._olivine = olivines.copy()
        self.offset_parameters = offset_parameters

        self._olivine_corrected = pd.Series(
            0.0,
            name="equilibration_crystallisation",
            index=inclusions.index,
            dtype=np.float16,
        )

        self._model_results = pd.Series(
            pd.NA,
            index=self.inclusions.index,
            name="isothermal_equilibration",
            dtype="boolean",
        )

        self.Kd_equilibrium = pd.Series(
            np.nan, name="Kd_equilibrium", index=self.inclusions.index
        )
        self.Kd_real = pd.Series(np.nan, name="Kd_real", index=self.inclusions.index)

    def reset(self):
        self.inclusions = self._inclusions_uncorrected.copy()
        self._olivine_corrected.loc[:] = 0.0
        self._model_results.loc[:] = pd.NA
        self.Kd_equilibrium.loc[:] = np.nan
        self.Kd_real.loc[:] = np.nan

    def _get_settings(self, **kwargs):

        stepsize = kwargs.get(
            "stepsize", getattr(PEC_configuration, "stepsize_equilibration")
        )

        self.stepsize = pd.Series(
            stepsize, index=self.inclusions.index, name="stepsize"
        )
        self.decrease_factor = getattr(PEC_configuration, "decrease_factor")
        self.dfO2 = model_configuration.dfO2
        self.Kd_converge = getattr(PEC_configuration, "Kd_converge")

    def _get_parameters(self, **kwargs):
        """Calculate melt liquidus temperature, fO2 and melt Fe3+/Fe2+ ratios"""

        melt = kwargs.get("melt_mol_fractions", self.inclusions.moles())
        pressure = kwargs.get("P_bar", self.P_bar)

        T_K = kwargs.get(
            "T_K",
            melt.temperature(
                pressure, offset=self.offset_parameters["temperature"], warn=False
            ),
        )
        fO2 = kwargs.get("fO2", calculate_fO2(T_K=T_K, P_bar=pressure))

        Fe3Fe2_model = Fe3Fe2_models_dict[model_configuration.Fe3Fe2_model]
        Fe3Fe2 = Fe3Fe2_model._calculate_Fe3Fe2_(
            melt_mol_fractions=melt,
            T_K=T_K,
            P_bar=pressure,
            fO2=fO2,
            offset_parameters=self.offset_parameters["Fe3Fe2"],
        )

        return T_K, fO2, Fe3Fe2

    def _get_FeMg_vector(self):

        # Fe-Mg exchange vectors
        FeMg_vector = pd.DataFrame(
            0, index=self.inclusions.index, columns=self.inclusions.columns
        )
        FeMg_vector.loc[:, ["FeO", "MgO"]] = [1.0, -1.0]

        return FeMg_vector

    def disequilibrium(self, Kd_equilibrium, Kd_real):

        disequilibrium = pd.Series(
            ~np.isclose(Kd_equilibrium, Kd_real, atol=self.Kd_converge, rtol=0),
            index=Kd_real.index,
        )

        return disequilibrium

    def calculate_Kds(self, **kwargs) -> Tuple[pd.Series, pd.Series]:
        """
        Model equilibrium Kds and calculate measured Kds
        """
        melt = kwargs.get("melt_mol_fractions", self.inclusions.moles())
        pressure = kwargs.get("P_bar", self.P_bar)
        forsterite = kwargs.get("forsterite", self._olivine.forsterite)

        T_K, fO2, Fe3Fe2 = self._get_parameters(melt_mol_fractions=melt, P_bar=pressure)

        Kd_equilibrium, Kd_observed = calculate_Kds(
            melt_mol_fractions=melt,
            forsterite=forsterite,
            T_K=T_K,
            P_bar=pressure,
            Fe3Fe2=Fe3Fe2,
            offset_parameters=self.offset_parameters["Kd"],
            fO2=fO2,
        )

        if isinstance(Kd_equilibrium, (float, int)):
            Kd_equilibrium = pd.Series(Kd_equilibrium, index=melt.index)

        return Kd_equilibrium, Kd_observed

    def isothermal_equilibration(self, var: Dict, olivine: pd.DataFrame):
        """
        Equilibration at given temperatures by crystallising or melting oliving.
        """

        # list of samples that did not equilibrate
        error_samples = []
        olivine_corrected = pd.Series(dtype=np.float16)
        melts_equilibrated = var["melt_mol_fractions"].copy()

        # Single core
        for sample, melt in var["melt_mol_fractions"].iterrows():

            olivine_amount, equilibration = isothermal_equilibration(
                melt_mol_fractions=melt,
                olivine_moles=olivine.loc[sample],
                T_K=var["T_K"].loc[sample],
                P_bar=var["P_bar"].loc[sample],
                temperature_offset=self.offset_parameters["temperature"],
            )

            self._model_results.loc[sample] = equilibration

            melts_equilibrated.loc[sample] = var["melt_mol_fractions"].loc[
                sample
            ] + olivine.loc[sample].mul(olivine_amount)
            # current iteration
            olivine_corrected.loc[sample] = np.float16(olivine_amount)
            # Running total

            if equilibration:
                continue

            error_samples.append(sample)
            olivine_corrected.loc[sample] = np.nan

        return melts_equilibrated, olivine_corrected, error_samples

    def _check_convergence(
        self, Kd_equilibrium_old, Kd_real_old, Kd_equilibrium_new, Kd_real_new, samples
    ):

        deltaKd_old = abs(Kd_equilibrium_old - Kd_real_old)
        deltaKd_new = abs(Kd_equilibrium_new - Kd_real_new)

        no_progress = np.logical_and(
            deltaKd_new > deltaKd_old, deltaKd_new > self.Kd_converge
        )
        error_samples = list(Kd_equilibrium_new.index[no_progress])

        # self._model_results.loc[error_samples] = False

        return error_samples

    def _check_progress(self, samples):

        try:
            stepsize_samples = self.stepsize.loc[samples]
            progress_errors = list(
                stepsize_samples.index[abs(stepsize_samples) < self._minimum_stepsize]
            )
            return progress_errors

        except AttributeError:
            w.warn("Model not yet initialised")

    def equilibrate(self, inplace=False, progressbar=True, **kwargs):
        """
        Equilibrate Fe and Mg between melt inclusions and their olivine host via diffusive Fe-Mg exchange
        """
        if not self._model_results.isna().all():
            self.reset()

        # Get settings
        self._get_settings(**kwargs)

        P_bar = self.P_bar

        # Calculate temperature and fO2
        T_K = self.inclusions.temperature(
            P_bar=P_bar, offset=self.offset_parameters["temperature"], warn=False
        )
        fO2 = calculate_fO2(T_K=T_K, P_bar=P_bar, dfO2=self.dfO2)
        # Get olivine forsterite contents
        forsterite_host = self._olivine.forsterite

        # Set up initial data
        melt_mol_fractions = self.inclusions.moles()
        Kd_equilibrium, Kd_real = self.calculate_Kds(
            melt_mol_fractions=melt_mol_fractions,
            T_K=T_K,
            P_bar=P_bar,
            fO2=fO2,
            forsterite=forsterite_host,
        )

        # Fe-Mg exchange vectors
        FeMg_vector = self._get_FeMg_vector()
        # Find disequilibrium inclusions
        disequilibrium = self.disequilibrium(
            Kd_equilibrium=Kd_equilibrium, Kd_real=Kd_real
        )

        self._model_results.loc[disequilibrium.index] = ~disequilibrium
        # Set stepsizes acoording to Kd disequilibrium
        self.stepsize.loc[Kd_real < Kd_equilibrium] = -self.stepsize.loc[
            Kd_real < Kd_equilibrium
        ].copy()
        # # Store olivine correction for the current iteration
        # olivine_corrected_loop = self._olivine_corrected.copy()

        ##### Main Fe-Mg exhange loop #####
        ###################################
        total_inclusions = melt_mol_fractions.shape[0]
        reslice = True

        variables = variables_container(
            melt_mol_fractions=melt_mol_fractions,
            forsterite=forsterite_host,
            T_K=T_K,
            P_bar=P_bar,
            FeMg_exchange=FeMg_vector,
            fO2=fO2,
            Kd_equilibrium=Kd_equilibrium,
            Kd_real=Kd_real,
            stepsize=self.stepsize,
        )

        if progressbar:
            bar_manager = alive_bar(
                total=total_inclusions,
                title=f"{'Equilibrating': <13} ...",
                manual=True,
                refresh_secs=1,
            )
        else:
            bar_manager = null_progressbar()

        i = 0  # loop iteration counter
        with bar_manager as bar:
            # , Pool() as pool
            bar(sum(~disequilibrium) / total_inclusions)
            while sum(disequilibrium) > 0:

                if reslice:
                    # Only reslice al the dataframes and series if the amount of disequilibrium inclusions has changed
                    samples = disequilibrium.index[disequilibrium]
                    variables.reslice(samples=samples)
                    var = variables.sliced

                # Exchange Fe and Mg
                var["melt_mol_fractions"] = var["melt_mol_fractions"].add(
                    var["FeMg_exchange"].mul(var["stepsize"], axis=0)
                )
                # Calculate new equilibrium Kd and Fe speciation
                var["Kd_equilibrium"], _ = self.calculate_Kds(
                    melt_mol_fractions=var["melt_mol_fractions"].normalise(),
                    **variables.fetch(("T_K", "fO2", "P_bar", "forsterite")),
                )

                *_, Fe3Fe2 = self._get_parameters(
                    melt_mol_fractions=var["melt_mol_fractions"].normalise(),
                    **variables.fetch(("T_K", "P_bar", "fO2")),
                )

                olivine = get_olivine_composition(
                    melt_mol_fractions=var["melt_mol_fractions"],
                    Fe3Fe2=Fe3Fe2,
                    Kd=var["Kd_equilibrium"],
                )

                # equilibrate the inclusions
                var["melt_mol_fractions"], olivine_corrected_loop, error_samples = (
                    self.isothermal_equilibration(var=var, olivine=olivine)
                )

                self._olivine_corrected.loc[
                    olivine_corrected_loop.index
                ] += olivine_corrected_loop

                # var["melt_mol_fractions"] = var["melt_mol_fractions"].normalise()
                ######################################################################
                # Recalculate Kds
                var["Kd_equilibrium"], var["Kd_real"] = self.calculate_Kds(
                    melt_mol_fractions=var["melt_mol_fractions"].normalise(),
                    **variables.fetch(("T_K", "P_bar", "fO2", "forsterite")),
                )
                # Find inclusions outside the equilibrium forsterite range
                disequilibrium_loop = self.disequilibrium(
                    Kd_equilibrium=var["Kd_equilibrium"], Kd_real=var["Kd_real"]
                )
                # Find overcorrected inclusions
                overstepped = ~np.equal(
                    np.sign(var["Kd_real"] - var["Kd_equilibrium"]),
                    np.sign(var["stepsize"]),
                )
                # Reverse one iteration and decrease stepsize for overcorrected inclusions
                decrease_stepsize = np.logical_and(overstepped, disequilibrium_loop)
                if sum(decrease_stepsize) > 0:
                    idx_stepsize = var["melt_mol_fractions"].index[decrease_stepsize]
                    var["melt_mol_fractions"].drop(
                        labels=idx_stepsize, axis=0, inplace=True
                    )
                    var["Kd_equilibrium"].drop(
                        labels=idx_stepsize, axis=0, inplace=True
                    )
                    var["Kd_real"].drop(labels=idx_stepsize, axis=0, inplace=True)
                    self._olivine_corrected.loc[
                        idx_stepsize
                    ] -= olivine_corrected_loop.loc[idx_stepsize]
                    self.stepsize.loc[idx_stepsize] = (
                        var["stepsize"].loc[idx_stepsize].div(self.decrease_factor)
                    )

                # Samples from the current loop that need to be updated
                samples_new = var["melt_mol_fractions"].index

                # Find which inclusions are not progressing towards Kd equilibrium
                convergence_errors = self._check_convergence(
                    Kd_equilibrium_old=Kd_equilibrium[samples_new],
                    Kd_real_old=Kd_real[samples_new],
                    Kd_equilibrium_new=var["Kd_equilibrium"],
                    Kd_real_new=var["Kd_real"],
                    samples=samples_new,
                )
                progress_errors = self._check_progress(samples=samples_new)

                # copy loop variables to the main variables
                melt_mol_fractions.loc[samples_new, :] = var[
                    "melt_mol_fractions"
                ].values

                Kd_equilibrium[samples_new] = var["Kd_equilibrium"].values
                Kd_real[samples_new] = var["Kd_real"].values

                disequilibrium_loop = self.disequilibrium(
                    Kd_equilibrium=var["Kd_equilibrium"], Kd_real=var["Kd_real"]
                )
                # self._model_results.loc[samples_new, "Kd_equilibration"] = (
                #     ~disequilibrium_loop
                # ).values

                disequilibrium.loc[samples_new] = disequilibrium_loop.values

                # Remove inclusions that returned equilibration errors from future iterations
                total_error_samples = (
                    error_samples + convergence_errors + progress_errors
                )
                disequilibrium.loc[total_error_samples] = False
                self._olivine_corrected.loc[total_error_samples] = np.nan
                melt_mol_fractions.loc[total_error_samples, :] = np.nan
                self._model_results.loc[total_error_samples] = False

                reslice = not disequilibrium.loc[samples].equals(disequilibrium_loop)

                if (i % 5 == 0) or (
                    sum(disequilibrium) == 0
                ):  # update progress bar every 5 iterations, crutch to fix IOStream flush error warnings.
                    bar(sum(~disequilibrium) / total_inclusions)
                i += 0

        corrected_compositions = melt_mol_fractions.wt_pc()

        # # Set compositions of inclusions with equilibration errors to NaNs.
        # idx_errors = self._olivine_corrected.index[self._olivine_corrected.isna()]
        # corrected_compositions.loc[idx_errors, :] = np.nan
        # self._model_results[idx_errors] = False

        # temperatures_new = corrected_compositions.temperature(P_bar=P_bar)
        self.inclusions = corrected_compositions
        self.Kd_equilibrium = Kd_equilibrium
        self.Kd_real = Kd_real

        if (n := (~self._model_results).sum()) > 0:
            w.warn(f"Isothermal equilibration not reached for {n} inclusions")

        if inplace:
            return

        return (
            corrected_compositions.copy(),
            self._olivine_corrected.copy(),
            self._model_results.copy(),
        )
