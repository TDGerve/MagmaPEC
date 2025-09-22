import warnings as w
from typing import Dict

import numpy as np
import pandas as pd
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

from MagmaPEC.equilibration_functions import diffusive_equilibration
from MagmaPEC.PEC_configuration import PEC_configuration
from MagmaPEC.tools import FeO_Target, null_progressbar, variables_container


class crystallisation_correction:
    """"""

    _minimum_stepsize = 1e-4

    def __init__(
        self, inclusions, olivines, P_bar, FeO_target: FeO_Target, offset_parameters
    ):

        self.inclusions = inclusions.copy()
        self._inclusions_uncorrected = self.inclusions.copy()
        self.P_bar = P_bar.copy()
        self._olivine = olivines.copy()
        self.FeO_target = FeO_target
        self.offset_parameters = offset_parameters

        self._olivine_corrected = pd.Series(
            0.0,
            name="PE_crystallisation",
            index=inclusions.index,
        )

        self._model_results = pd.DataFrame(
            {"Kd_equilibration": pd.NA, "FeO_converge": pd.NA},
            index=self.inclusions.index,
            dtype="boolean",
        )

    def reset(self):
        self.inclusions = self._inclusions_uncorrected.copy()
        self._olivine_corrected.loc[:] = np.nan
        self._model_results.loc[:, :] = pd.NA

    def _get_settings(self, **kwargs):

        # Get settings
        stepsize = kwargs.get(
            "stepsize", getattr(PEC_configuration, "stepsize_crystallisation")
        )
        self.stepsize = pd.Series(
            stepsize, index=self.inclusions.index, name="stepsize"
        )
        self.decrease_factor = getattr(PEC_configuration, "decrease_factor")
        self.FeO_converge = kwargs.get(
            "FeO_converge", getattr(PEC_configuration, "FeO_converge")
        )

    def _get_FeMg_vector(self):

        # Fe-Mg exchange vectors
        FeMg_vector = pd.Series(0, index=self.inclusions.columns, name="FeMg_exchange")
        FeMg_vector.loc[["FeO", "MgO"]] = [1, -1]

        return FeMg_vector

    def FeO_mismatch(self, FeO, FeO_target):

        FeO_mismatch = pd.Series(
            ~np.isclose(FeO, FeO_target, atol=self.FeO_converge, rtol=0),
            index=FeO.index,
            name="FeO_mismatch",
        )

        return FeO_mismatch

    def Kd_equilibration(self, var: Dict, FeMg_vector):

        error_samples = []
        melts_equilibrated = var["melt_mol_fractions"].copy()

        for sample, melt in var["melt_mol_fractions"].iterrows():

            exchange_amount, equilibration = diffusive_equilibration(
                melt_mol_fractions=melt,
                forsterite=var["olivine"].forsterite.loc[sample],
                FeMg_vector=FeMg_vector,
                P_bar=var["P_bar"].loc[sample],
                offset_parameters=self.offset_parameters,
            )

            self._model_results.loc[sample, "Kd_equilibration"] = equilibration

            melts_equilibrated.loc[sample] = var["melt_mol_fractions"].loc[
                sample
            ] + FeMg_vector.mul(exchange_amount)

            self._model_results.loc[sample, "Kd_equilibration"] = equilibration

            if equilibration:
                continue
            # Catch inclusions that cannot reach Kd equilibrium
            error_samples.append(sample)
            self._olivine_corrected.loc[sample] = np.nan

        return melts_equilibrated.normalise(), error_samples

    def correct(
        self,
        equilibration_crystallisation=0.0,
        inplace=False,
        progressbar=True,
        **kwargs,
    ):
        """
        Correct an olivine hosted melt inclusion for post entrapment crystallisation or melting by
        respectively melting or crystallising host olivine.
        Expects the melt inclusion is completely equilibrated with the host crystal.
        The models exits when the user input original melt inclusion FeO content is reached.
        Loosely based on the postentrapment reequilibration procedure in Petrolog:

        L. V. Danyushesky and P. Plechov (2011)
        Petrolog3: Integrated software for modeling crystallization processes
        Geochemistry, Geophysics, Geosystems, vol 12
        """
        if not self._model_results.isna().all().all():
            self.reset()

        self._get_settings()

        # Inclusion compositions in oxide mol fractions
        melt_mol_fractions = self.inclusions.moles()
        # Convert to the total amount of moles after equilibration
        melt_mol_fractions = melt_mol_fractions.mul(
            (1 + equilibration_crystallisation),
            axis=0,
        )

        FeMg_vector = self._get_FeMg_vector()
        # Starting FeO and temperature
        FeO = self.inclusions["FeO"].copy()
        FeO_target = self.FeO_target.target(melt_wtpc=self.inclusions)

        self.stepsize.loc[FeO > FeO_target] = -self.stepsize.loc[FeO > FeO_target]
        FeO_mismatch = self.FeO_mismatch(FeO=FeO, FeO_target=FeO_target)
        self._model_results.loc[FeO_mismatch.index, "FeO_converge"] = ~FeO_mismatch

        ##### OLIVINE MELTING/CRYSTALLISATION LOOP #####
        reslice = True
        total_inclusions = melt_mol_fractions.shape[0]

        variables = variables_container(
            melt_mol_fractions=melt_mol_fractions,
            olivine=self._olivine,
            stepsize=self.stepsize,
            P_bar=self.P_bar,
        )
        if progressbar:
            bar_manager = alive_bar(
                total=total_inclusions,
                title=f"{'Correcting': <13} ...",
                manual=True,
                refresh_secs=1,
            )
        else:
            bar_manager = null_progressbar()

        i = 0  # loop iteration counter
        with bar_manager as bar:

            while sum(FeO_mismatch) > 0:

                bar(sum(~FeO_mismatch) / total_inclusions)

                if reslice:
                    samples = FeO_mismatch.index[FeO_mismatch]
                    variables.reslice(samples=samples)
                    var = variables.sliced

                var["melt_mol_fractions"] = var["melt_mol_fractions"] + var[
                    "olivine"
                ].mul(var["stepsize"], axis=0)

                # var["melt_mol_fractions"] = var["melt_mol_fractions"].normalise()
                self._olivine_corrected.loc[samples] += var["stepsize"]
                #################################################
                ##### Exchange Fe-Mg to keep Kd equilibrium #####
                var["melt_mol_fractions"], Kd_equilibration_error = (
                    self.Kd_equilibration(var=var, FeMg_vector=FeMg_vector)
                )

                #################################################
                # Recalculate FeO
                melt_wtpc_loop = var["melt_mol_fractions"].wt_pc()
                FeO = melt_wtpc_loop["FeO"]

                FeO_target_loop = self.FeO_target.target(melt_wtpc=melt_wtpc_loop)

                # Find FeO mismatched and overcorrected inclusions
                FeO_mismatch_loop = self.FeO_mismatch(
                    FeO=FeO, FeO_target=FeO_target_loop
                )

                FeO_overstepped = ~np.equal(
                    np.sign(FeO_target_loop - FeO), np.sign(var["stepsize"])
                )
                decrease_stepsize_FeO = np.logical_and(
                    FeO_overstepped, FeO_mismatch_loop
                )

                if sum(decrease_stepsize_FeO) > 0:
                    # Reverse one step and decrease stepsize
                    reverse_FeO = var["melt_mol_fractions"].index[decrease_stepsize_FeO]
                    var["melt_mol_fractions"].drop(
                        labels=reverse_FeO, axis=0, inplace=True
                    )
                    self._olivine_corrected.loc[reverse_FeO] -= var["stepsize"].loc[
                        reverse_FeO
                    ]
                    self.stepsize.loc[reverse_FeO] = (
                        var["stepsize"].loc[reverse_FeO].div(self.decrease_factor)
                    )

                # determine FeO convergence
                melt_wtpc_loop = var["melt_mol_fractions"].wt_pc()
                FeO = melt_wtpc_loop["FeO"]
                FeO_target = self.FeO_target.target(melt_wtpc=melt_wtpc_loop)
                FeO_mismatch_loop = self.FeO_mismatch(FeO=FeO, FeO_target=FeO_target)

                # Copy loop data to the main variables
                melt_mol_fractions.loc[var["melt_mol_fractions"].index, :] = var[
                    "melt_mol_fractions"
                ].copy()
                # mi_wtPercent = mi_moles.wt_pc()

                FeO_mismatch.loc[FeO_mismatch_loop.index] = FeO_mismatch_loop.values

                reslice = not FeO_mismatch.loc[samples].equals(FeO_mismatch_loop)

                # Remove inclusions with Kd equilibration or FeO convergence errors from future iterations
                FeO_converge_error = list(
                    self.stepsize.index[abs(self.stepsize) < self._minimum_stepsize]
                )  # Stop iterating if stepsize falls below 1e-5
                self._model_results.loc[FeO_converge_error, "FeO_converge"] = False
                remove_samples = Kd_equilibration_error + FeO_converge_error
                FeO_mismatch[remove_samples] = False
                self._olivine_corrected.loc[remove_samples] = np.nan

                if (i % 5 == 0) or (
                    sum(FeO_mismatch) == 0
                ):  # update progress bar every 5 iterations, crutch to fix IOStream flush error warnings.
                    bar(sum(~FeO_mismatch) / total_inclusions)
                i += 1

        self._model_results.loc[FeO_mismatch.index, "FeO_converge"] = ~FeO_mismatch
        self.inclusions.loc[melt_mol_fractions.index] = (
            melt_mol_fractions.wt_pc().values
        )

        # Set non-corrected inclusions to NA.
        idx_errors = self._olivine_corrected.index[self._olivine_corrected.isna()]
        self.inclusions.loc[idx_errors] = np.nan
        self._model_results.loc[idx_errors, "FeO_converge"] = pd.NA
        self._olivine_corrected.loc[idx_errors] = np.nan

        if (n := (~self._model_results["FeO_converge"]).sum()) > 0:
            w.warn(f"FeO not converged for {n} inclusions")

        if inplace:
            return

        return (
            self.inclusions.copy(),
            self._olivine_corrected.copy(),
            self._model_results.copy(),
        )
