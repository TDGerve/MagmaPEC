from typing import Tuple, Union

import numpy as np
import pandas as pd
from MagmaPandas.fO2.fO2_calculate import calculate_fO2
from MagmaPandas.MagmaFrames import Melt, Olivine

from MagmaPEC.Kd_calculation import calculate_observed_Kd
from MagmaPEC.PEC_configuration import PEC_configuration
from MagmaPEC.PEC_model.scalar import (
    crystallisation_correction_scalar,
    equilibration_scalar,
)
from MagmaPEC.PEC_model.vector import crystallisation_correction, equilibration
from MagmaPEC.plots import PEC_plot
from MagmaPEC.tools import FeO_Target


class PEC:
    """
    Class for post-entrapment crystallisation (PEC) correction of olivine-hosted melt inclusions. The algorithm is modified after Petrolog3:

    L. V. Danyushesky and P. Plechov (2011) Petrolog3: Integrated software for modeling crystallization processes Geochemistry, Geophysics, Geosystems, vol 12

    Model specific settings like calculation stepsizes and convergence values are controlled by :py:class:`~MagmaPEC.PEC_configuration.PEC_configuration`.

    More general settings like |fO2| buffer (offsets) and model selection for Kd, |Fe3Fe2| and liquidus temperature are set in the configuration of `MagmaPandas <https://magmapandas.readthedocs.io/en/latest/notebooks/config.html>`_

    Parameters
    ----------
    inclusions : :py:class:`~magmapandas:MagmaPandas.MagmaFrames.melt.Melt`
        melt inclusion compositions in oxide wt. %
    olivines : :py:class:`~magmapandas:MagmaPandas.MagmaFrames.olivine.Olivine`, :py:class:`Pandas Series <pandas:pandas.Series>`, float
        Olivine compositions in oxide wt. % as Olivine MagmaFrame, or olivine forsterite contents as pandas Series or float.
    P_bar : float, :py:class:`Pandas Series <pandas:pandas.Series>`
        Pressures in bar
    FeO_target : float, :py:class:`Pandas Series <pandas:pandas.Series>`, Callable
        Melt inclusion initial FeO content as a fixed value for all inclusions, inclusion specific values, or a predictive equation based on melt composition. The callable needs to accept a :py:class:`Pandas DataFrame <pandas:pandas.DataFrame>` with melt compositions in oxide wt. % as input and return an array-like object with initial FeO contents per inclusion.
    Fe3Fe2_offset_parameters : float, array-like
        offsets of calculated melt Fe3+/Fe2+ ratios in standard deviations.
    Kd_offset_parameters : float, array-like
        offsets of calculated olivine-melt Fe-Mg Kd's in standard deviations.
    temperature_offset_parameters : float, array-like
        offsets of calculated temperatures in standard deviations.

    Attributes
    ----------
    inclusions : :py:class:`~magmapandas:MagmaPandas.MagmaFrames.melt.Melt`
        Melt inclusion compositions in oxide wt. %. Compositions are updated during the PEC correction procedure
    P_bar : :py:class:`Pandas Series <pandas:pandas.Series>`
        Pressures in bar
    FeO_target : float, :py:class:`Pandas Series <pandas:pandas.Series>`, Callable
        Melt inclusion initial FeO contents.
    olivine_corrected : :py:class:`Pandas DataFrame <pandas:pandas.DataFrame>`
        Dataframe with columns *equilibration_crystallisation*, *PE_crystallisation* and *total_crystallisation*, storing total PEC and crystallisation amounts during the equilibration and crystallisation stages.


    """

    def __init__(
        self,
        inclusions: Melt,
        olivines: Union[Olivine, pd.Series, float],
        P_bar: Union[float, int, pd.Series],
        FeO_target: Union[float, pd.Series, callable],
        Fe3Fe2_offset_parameters: float = 0.0,
        Kd_offset_parameters: float = 0.0,
        temperature_offset_parameters: float = 0.0,
        **kwargs,
    ):

        self.offset_parameters = {
            "Fe3Fe2": Fe3Fe2_offset_parameters,
            "Kd": Kd_offset_parameters,
            "temperature": temperature_offset_parameters,
        }

        self._FeO_as_function = False

        # Process attributes
        ######################

        # inclusions
        if not isinstance(inclusions, Melt):
            raise TypeError("Inclusions is not a Melt MagmaFrame")
        else:
            inclusions = inclusions.fillna(0.0)
            self.inclusions = inclusions.normalise()
            self.inclusions_uncorrected = self.inclusions.copy()

        # olivines
        self._olivine = self._process_olivines(
            olivines=olivines, samples=inclusions.index
        )
        self._olivine = self._olivine.reindex(
            columns=self.inclusions.columns, fill_value=0.0
        ).recalculate()

        # pressure
        try:
            if len(P_bar) != self.inclusions.shape[0]:
                raise ValueError(
                    "Number of pressure inputs and melt inclusions does not match"
                )
        except TypeError:
            pass
        if hasattr(P_bar, "index"):
            try:
                P_bar = P_bar.loc[inclusions.index]
            except KeyError as err:
                print("Inclusion and P_bar indeces don't match")
                raise err
            self.P_bar = P_bar
        else:
            self.P_bar = pd.Series(P_bar, index=self.inclusions.index, name=P_bar)

        # FeO target
        self.FeO_target = FeO_Target(
            FeO_target=FeO_target, samples=self.inclusions.index
        )

        self._create_output_dataframes(samples=self.inclusions.index)

    def reset(self):
        """
        Reset all data to their initial, uncorrected states.
        """
        self.inclusions = self.inclusions_uncorrected.copy()
        self._olivine_corrected.loc[:, :] = np.nan
        self._model_results.loc[:, :] = pd.NA

    @property
    def olivine_corrected(self):
        return self._olivine_corrected.mul(100)

    @olivine_corrected.setter
    def olivine_corrected(self, value):
        print("read only")

    @property
    def olivine(self):
        return self._olivine.wt_pc()

    @olivine.setter
    def olivine(self, value):
        print("olivine is read only")

    @property
    def Fe_loss(self) -> pd.Series:
        """
        Booleans set to True for inclusions that have experienced Fe-loss
        """
        FeO_converge = getattr(PEC_configuration, "FeO_converge")
        FeO_target = self.FeO_target.target(melt_wtpc=self.inclusions)
        return pd.Series(
            ~np.isclose(
                FeO_target,
                self.inclusions["FeO"],
                atol=FeO_converge,
                rtol=0,
            ),
            index=self.inclusions.index,
            name="Fe_loss",
        )

    def equilibrate_inclusions(self, progressbar=True, **kwargs):
        """
        Run correction stage 1.

        Fe-Mg equilibrium between inclusions and olivine hosts is restored via isothermal Fe-Mg cation exchange. Equilibrium is checked with modelled partitioning coefficients (Kd).

        progressbar :   boolean
            show progress bar.
        """

        model = equilibration(
            inclusions=self.inclusions,
            olivines=self._olivine,
            P_bar=self.P_bar,
            offset_parameters=self.offset_parameters,
        )

        corrected_melt_compositions, olivine_crystallised, model_results = (
            model.equilibrate(inplace=False, progressbar=progressbar, **kwargs)
        )

        self.inclusions = corrected_melt_compositions
        self._olivine_corrected["equilibration_crystallisation"] = (
            olivine_crystallised.values
        )
        self._model_results["isothermal_equilibration"] = model_results.values

    def correct_olivine_crystallisation(
        self, inplace=False, progressbar=True, **kwargs
    ):
        """
        Run correction stage 2.

        Correct melt inclusions for post entrapment modification by melting or crystallising host olivine.
        Expects complete Fe-Mg equilibration between inclusions and host crystals (i.e. the output from stage 1: :py:meth:`~MagmaPEC.PEC_model.PEC.equilibrate_inclusions`).
        The models exits when inclusion initial FeO contents are restored.

        progressbar :   boolean
            show progress bar.
        """

        if not self._model_results["isothermal_equilibration"].any():
            raise RuntimeError("None of the inclusions are equilibrated")

        # only select samples that reached isothermal equilibrium.
        select_samples = self._model_results["isothermal_equilibration"]

        model = crystallisation_correction(
            inclusions=self.inclusions.loc[select_samples],
            olivines=self._olivine.loc[select_samples],
            P_bar=self.P_bar.loc[select_samples],
            FeO_target=self.FeO_target,
            offset_parameters=self.offset_parameters,
        )

        corrected_melt_compositions, olivine_crystallised, model_results = (
            model.correct(
                equilibration_crystallisation=self._olivine_corrected.loc[
                    select_samples, "equilibration_crystallisation"
                ],
                inplace=False,
                progressbar=progressbar,
            )
        )

        samples = corrected_melt_compositions.index
        self.inclusions.loc[samples] = corrected_melt_compositions.values
        self._olivine_corrected.loc[samples, "PE_crystallisation"] = (
            olivine_crystallised.values
        )
        self._model_results.loc[samples, ["Kd_equilibration", "FeO_converge"]] = (
            model_results.values
        )

    def correct(
        self, progressbar=True, **kwargs
    ) -> Tuple[Melt, pd.DataFrame, pd.DataFrame]:
        """
        Correct inclusions for PEC.

        Runs Stage 1, :py:meth:`~MagmaPEC.PEC_model.PEC.equilibrate_inclusions`, and 2, :py:meth:`~MagmaPEC.PEC_model.PEC.correct_olivine_crystallisation`, to fully correct melt inclusion for post-entrapment modification processes.

        progressbar :   boolean
            show progress bar.
        """

        self.equilibrate_inclusions(progressbar=progressbar, **kwargs)
        self.correct_olivine_crystallisation(progressbar=progressbar, **kwargs)

        self._olivine_corrected["total_crystallisation"] = self._olivine_corrected[
            ["equilibration_crystallisation", "PE_crystallisation"]
        ].sum(axis=1, skipna=False)

        return (
            self.inclusions.copy(),
            self._olivine_corrected.mul(100),
            self._model_results.copy(),
        )

    def get_PTX(self, P_bar=None):
        """
        P_bar   : array-like
            pressures in bar for each inclusion.

        Returns
        -------
            pandas dataframe with pressures, temperatures, Kd (current inclusion value, not modelled), melt Fe3Fe2, and fO2 for uncorrected and corrected inclusions.
        """

        if (not hasattr(self, "_olivine_corrected")) or (
            self._olivine_corrected.isna().all().all()
        ):
            return self._get_PTX(inclusions=self.inclusions, P_bar=P_bar)

        uncorrected = self._get_PTX(inclusions=self.inclusions_uncorrected, P_bar=P_bar)
        corrected = self._get_PTX(inclusions=self.inclusions, P_bar=P_bar)

        uncorrected.columns = pd.MultiIndex.from_product(
            [uncorrected.columns, ["uncorrected"]]
        )
        corrected.columns = pd.MultiIndex.from_product(
            [corrected.columns, ["corrected"]]
        )

        return pd.concat([uncorrected, corrected], axis=1).sort_index(axis=1)

    def _get_PTX(self, inclusions, P_bar=None):

        if P_bar is None:
            P_bar, T_K = self._PT_iterate(inclusions=inclusions)
        else:
            T_K = inclusions.temperature(P_bar=P_bar)

        Fe3Fe2 = inclusions.Fe3Fe2(T_K=T_K, P_bar=P_bar)

        forsterite = self.olivine.forsterite
        # Kd_modelled = inclusions.Kd_olivine_FeMg_eq(
        #     T_K=T_K, P_bar=P_bar, Fe3Fe2=Fe3Fe2, forsterite_initial=forsterite
        # )
        Kd_measured = calculate_observed_Kd(
            inclusions.moles(), Fe3Fe2=Fe3Fe2, forsterite=forsterite
        )
        fO2 = calculate_fO2(T_K=T_K, P_bar=P_bar)

        return pd.DataFrame(
            {
                "P_bar": P_bar,
                "T_K": T_K,
                # "Kd_modelled": Kd_modelled,
                "Kd": Kd_measured,
                "Fe3Fe2": Fe3Fe2,
                "fO2_Pa": fO2,
            }
        )

    def _PT_iterate(self, inclusions, convergence=0.01):

        T_K = inclusions.temperature(P_bar=1e3)
        P_bar = inclusions.volatile_saturation_pressure(T_K=T_K)

        while True:
            T_K = inclusions.temperature(P_bar=P_bar)
            P_bar_new = inclusions.volatile_saturation_pressure(T_K=T_K)
            dP = (P_bar_new - P_bar) / P_bar_new
            P_bar = P_bar_new.copy()
            if (dP < convergence).all():
                break

        return P_bar, T_K

    def correct_inclusion(
        self, index, plot=True, intermediate_steps=False, **kwargs
    ) -> pd.DataFrame:
        """Correct a single inclusion for PEC

        Parameters
        ----------
        index               :   int, str
            row index of the inclusion in the DataFrame as integer, or name of the inclusion as string
        plot                :   boolean
            print MgO vs FeO plot of the PEC results if True
        intermediate_steps  :   boolean
            keep intermediate steps C1A and C2A if True
        """

        if type(index) == int:
            index = self.inclusions_uncorrected.index[index]

        inclusion = self.inclusions_uncorrected.loc[index].copy().squeeze()
        olivine = self._olivine.loc[index].copy().squeeze()
        FeO_target = self.FeO_target.target(melt_wtpc=inclusion)
        P_bar = self.P_bar.loc[index].squeeze()

        if self.FeO_target._as_function:
            FeO_target = self.FeO_target.function

        equilibrated, olivine_equilibrated, *_ = equilibration_scalar(
            inclusion, olivine, P_bar, intermediate_steps=intermediate_steps, **kwargs
        )
        corrected, olivine_corrected, *_ = crystallisation_correction_scalar(
            equilibrated.iloc[-1].copy(),
            olivine,
            FeO_target,
            P_bar,
            eq_crystallised=olivine_equilibrated[-1],
            intermediate_steps=intermediate_steps,
            **kwargs,
        )
        total_corrected = olivine_corrected[-1]

        equilibrated["correction"] = "equilibration"
        corrected["correction"] = "correction"

        total_inclusion = pd.concat([equilibrated, corrected.iloc[1:]], axis=0)

        if plot:

            PEC_plot(
                name=index,
                equilibration=equilibrated,
                correction=corrected,
                FeO_target=self.FeO_target,
                PEC_amount=total_corrected,
            )

        return total_inclusion, olivine_corrected

    def _process_olivines(self, olivines, samples: pd.Index):

        try:
            if not olivines.index.equals(samples):
                raise IndexError("olivine and inclusion indeces do not match")
        except AttributeError:
            pass

        if isinstance(olivines, Olivine):
            olivines = olivines.fillna(0.0)
            return olivines.moles()
        # For olivines

        try:
            if len(olivines) != len(samples):
                raise ValueError("Number of olivines and inclusions does not match")
        except TypeError:
            pass
        forsterite = pd.Series(olivines, index=samples, name="forsterite")
        if (~forsterite.between(0, 1)).any():
            raise ValueError(
                "olivine host forsterite contents are not all between 0 and 1"
            )
        olivine = Olivine(
            {"MgO": forsterite * 2, "FeO": (1 - forsterite) * 2, "SiO2": 1},
            index=samples,
            units="mol fraction",
            datatype="oxide",
        )
        return olivine.normalise()

    def _create_output_dataframes(self, samples: pd.DataFrame):

        self._olivine_corrected = pd.DataFrame(
            np.nan,
            columns=[
                "equilibration_crystallisation",
                "PE_crystallisation",
                "total_crystallisation",
            ],
            index=samples,
        )

        self._model_results = pd.DataFrame(
            {
                "isothermal_equilibration": pd.NA,
                "Kd_equilibration": pd.NA,
                "FeO_converge": pd.NA,
            },
            index=samples,
            dtype="boolean",
        )
