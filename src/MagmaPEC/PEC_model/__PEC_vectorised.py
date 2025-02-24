import warnings as w
from typing import Tuple, Union

import numpy as np
import pandas as pd

# Progress bar for lengthy calculations
from alive_progress import alive_bar, config_handler
from scipy.optimize import root_scalar

config_handler.set_global(
    title_length=17,
    manual=True,
    theme="smooth",
    spinner=None,
    stats=False,
    length=30,
    force_tty=True,
)


from MagmaPandas.configuration import configuration
from MagmaPandas.Fe_redox.Fe3Fe2_models import Fe3Fe2_models_dict
from MagmaPandas.fO2 import calculate_fO2
from MagmaPandas.Kd.Ol_melt import Kd_olmelt_FeMg_models_dict
from MagmaPandas.MagmaFrames import Melt, Olivine

from MagmaPEC.Kd_calculation import calculate_observed_Kd
from MagmaPEC.PEC_configuration import PEC_configuration
from MagmaPEC.PEC_model.scalar.Crystallisation_correction_scalar import (
    crystallisation_correction,
)
from MagmaPEC.PEC_model.scalar.Equilibration_scalar import equilibration_scalar

# from .PEC_functions import _root_Kd, _root_temperature


class PEC_olivine:
    # TODO link to configuration documentation.
    """
    Class for post-entrapment crystallisation (PEC) correction of olivine-hosted melt inclusions.

    Model settings are controlled by the global PEC configuration.

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
        self._Fe3Fe2_offset_parameters = Fe3Fe2_offset_parameters
        self._Kd_offset_parameters = Kd_offset_parameters
        self._temperature_offset_parameters = temperature_offset_parameters

        self._olivine_corrected = pd.DataFrame(
            0.0,
            columns=[
                "equilibration_crystallisation",
                "PE_crystallisation",
                "total_crystallisation",
            ],
            index=inclusions.index,
        )
        self._FeO_as_function = False

        # Process attributes
        ######################

        # For inclusions
        if not isinstance(inclusions, Melt):
            raise TypeError("Inclusions is not a Melt MagmaFrame")
        else:
            inclusions = inclusions.fillna(0.0)
            self.inclusions = inclusions.normalise()
            self.inclusions_uncorrected = self.inclusions.copy()
        # For olivine
        if hasattr(olivines, "index"):
            try:
                olivines = olivines.loc[inclusions.index]
            except KeyError as err:
                print("Inclusion and olivine indeces don't match")
                raise err

        # For olivines
        if not isinstance(olivines, Olivine):
            try:
                if len(olivines) != self.inclusions.shape[0]:
                    raise ValueError("Number of olivines and inclusions does not match")
            except TypeError:
                pass
            forsterite = pd.Series(
                olivines, index=self.inclusions.index, name="forsterite"
            )
            if (~forsterite.between(0, 1)).any():
                raise ValueError(
                    "olivine host forsterite contents are not all between 0 and 1"
                )
            olivine = Olivine(
                {"MgO": forsterite * 2, "FeO": (1 - forsterite) * 2, "SiO2": 1},
                index=self.inclusions.index,
                units="mol fraction",
                datatype="oxide",
            )
            self._olivine = olivine.normalise()
        else:
            olivines = olivines.fillna(0.0)
            self._olivine = olivines.moles()
        self._olivine = self._olivine.reindex(
            columns=self.inclusions.columns, fill_value=0.0
        )

        # For pressure
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

        # For FeO
        try:
            if len(FeO_target) != len(self.inclusions):
                raise ValueError(
                    "Number of initial FeO inputs and inclusions does not match"
                )
            if hasattr(FeO_target, "index"):
                if not FeO_target.equals(self.inclusions.index):
                    raise ValueError(
                        "FeO target inputs and inclusion indeces don't match"
                    )
                self.FeO_target = FeO_target
            else:
                self.FeO_target = pd.Series(
                    FeO_target, index=self.inclusions.index, name="FeO_target"
                )
        except TypeError:
            if isinstance(FeO_target, (int, float)):
                self.FeO_target = pd.Series(
                    FeO_target, index=self.inclusions.index, name="FeO_target"
                )
            elif hasattr(FeO_target, "__call__"):
                self._FeO_as_function = True
                self.FeO_function = FeO_target
                self.FeO_target = self.FeO_function(self.inclusions)

        self._model_results = pd.DataFrame(
            {
                "isothermal_equilibration": False,
                "Kd_equilibration": False,
                "FeO_converge": True,
            },
            index=self.inclusions.index,
            dtype=bool,
        )

    def reset(self):
        self.inclusions = self.inclusions_uncorrected.copy()
        self._olivine_corrected = pd.DataFrame(
            0.0,
            columns=[
                "equilibration_crystallisation",
                "PE_crystallisation",
                "total_crystallisation",
            ],
            index=self.inclusions.index,
        )
        if self._FeO_as_function:
            self.FeO_target = self.FeO_function(self.inclusions)

    @property
    def olivine_corrected(self):
        return self._olivine_corrected.mul(100)

    @olivine_corrected.setter
    def olivine_corrected(self, value):
        print("read only!")

    @property
    def olivine(self):
        return self._olivine.wt_pc()

    @olivine.setter
    def olivine(self, value):
        print("Olivine is read only")

    @property
    def equilibrated(self) -> pd.Series:
        """
        Booleans indicating which inclusions are in equilibrium with their host olivine
        based on modelled and observed Fe-Mg exchange Kd's.
        """

        Kd_converge = getattr(PEC_configuration, "Kd_converge") * 1.5
        Kd_equilibrium, Kd_real = self.calculate_Kds()

        return pd.Series(
            np.isclose(Kd_equilibrium, Kd_real, atol=Kd_converge, rtol=0),
            index=Kd_equilibrium.index,
            name="equilibrated",
        )

    @property
    def Fe_loss(self) -> pd.Series:
        """
        Booleans indicating which inclusions have experienced Fe loss
        """
        FeO_converge = getattr(PEC_configuration, "FeO_converge")
        if self._FeO_as_function:
            FeO_target = self.FeO_function(self.inclusions)
        else:
            FeO_target = self.FeO_target
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

    def calculate_Fe3Fe2(self, **kwargs) -> pd.Series:
        """
        Calculate melt inclusion Fe3+/Fe2+ ratios

        Returns
        -------
        Fe3Fe2 : pd.Series
            melt inclusion Fe3+/Fe2+ ratios.
        """

        Fe3Fe2_model = Fe3Fe2_models_dict[configuration.Fe3Fe2_model]

        melt = kwargs.get("melt_mol_fractions", self.inclusions.moles())
        melt_wtpc = melt.wt_pc()

        pressure = kwargs.get("P_bar", self.P_bar)
        dfO2 = kwargs.get("dfO2", configuration.dfO2)

        T_K = kwargs.get(
            "T_K",
            melt_wtpc.temperature(
                pressure, offset=self._temperature_offset_parameters, warn=False
            ),
        )
        fO2 = kwargs.get("fO2", calculate_fO2(T_K=T_K, P_bar=pressure, dfO2=dfO2))

        Fe3Fe2 = Fe3Fe2_model.calculate_Fe3Fe2(
            melt_mol_fractions=melt, T_K=T_K, fO2=fO2, P_bar=pressure
        )
        # if self._Fe3Fe2_offset_parameters != 0.0:
        offset = Fe3Fe2_model.get_offset(
            melt_composition=melt,
            Fe3Fe2=Fe3Fe2,
            offset_parameters=self._Fe3Fe2_offset_parameters,
            pressure=pressure,
        )

        Fe3Fe2 = Fe3Fe2 + offset
        try:
            Fe3Fe2[Fe3Fe2 <= 0.0] = 1e-6
        except TypeError:
            Fe3Fe2 = 1e-6 if Fe3Fe2 <= 0 else Fe3Fe2

        try:
            if len(Fe3Fe2) == 1:
                return np.array(Fe3Fe2)[0]
            return Fe3Fe2

        except TypeError:
            return Fe3Fe2

    @staticmethod
    def calculate_Fe2FeTotal(Fe3Fe2, **kwargs):

        return 1 / (1 + Fe3Fe2)

    def calculate_Kds(self, **kwargs) -> Tuple[pd.Series, pd.Series]:
        """
        Model equilibrium Kds and calculate measured Kds
        """
        Kd_model = Kd_olmelt_FeMg_models_dict[configuration.Kd_model]

        dfO2 = kwargs.get("dfO2", configuration.dfO2)

        melt = kwargs.get("melt", self.inclusions.moles())
        pressure = kwargs.get("P_bar", self.P_bar)
        forsterite = kwargs.get("forsterite", self._olivine.forsterite)

        T_K = kwargs.get(
            "T_K",
            melt.temperature(
                pressure, offset=self._temperature_offset_parameters, warn=False
            ),
        )
        fO2 = kwargs.get("fO2", calculate_fO2(T_K=T_K, P_bar=pressure, dfO2=dfO2))

        Fe3Fe2 = self.calculate_Fe3Fe2(
            melt_mol_fractions=melt, P_bar=pressure, dfO2=dfO2, fO2=fO2, T_K=T_K
        )

        Kd_observed = calculate_observed_Kd(
            melt_mol_fractions=melt, Fe3Fe2=Fe3Fe2, forsterite=forsterite
        )
        # Fe2_FeTotal = 1 / (1 + Fe3Fe2)
        # melt_MgFe = melt["MgO"] / (melt["FeO"] * Fe2_FeTotal)
        # olivine_MgFe = forsterite / (1 - forsterite)
        # Kd_observed = melt_MgFe / olivine_MgFe

        if isinstance(Kd_observed, pd.Series):
            Kd_observed.rename("real", inplace=True)
            Kd_observed.index.name = "name"

        Kd_equilibrium = Kd_model.calculate_Kd(
            melt_mol_fractions=melt,
            T_K=T_K,
            P_bar=pressure,
            Fe3Fe2=Fe3Fe2,
            forsterite_intial=forsterite,
        )

        if self._Kd_offset_parameters != 0.0:
            offset = Kd_model.get_offset(
                melt_composition=melt,
                offset_parameters=self._Kd_offset_parameters,
            )
            Kd_equilibrium = Kd_equilibrium + offset

        try:
            Kd_equilibrium[Kd_equilibrium <= 0.0] = 1e-6
        except TypeError:
            Kd_equilibrium = 1e-6 if Kd_equilibrium <= 0.0 else Kd_equilibrium

        # if isinstance(Kd_equilibrium, (float, int)):
        #     Kd_equilibrium = pd.Series(Kd_equilibrium, index=melt.index)
        # Kd_equilibrium.rename("equilibrium", inplace=True)
        # Kd_equilibrium.index.name = "name"
        if isinstance(Kd_equilibrium, pd.Series):
            Kd_equilibrium.rename("equilibrium", inplace=True)
            Kd_equilibrium.index.name = "name"

        return Kd_equilibrium, Kd_observed

    def _crystallise_multicore(self, sample):
        """
        Doesn't seem to work
        """

        name, inclusion, olivine, temperature, pressure = sample

        temperature_new = inclusion.temperature(
            pressure, offset=self._temperature_offset_parameters, warn=False
        )

        sign = np.sign(temperature - temperature_new)

        olivine_amount = root_scalar(
            self._root_temperature,
            args=(
                inclusion,
                olivine,
                temperature,
                pressure,
            ),
            x0=0,
            x1=sign * 0.01,
        ).root

        return name, olivine_amount

    def _root_temperature(
        self, olivine_amount, melt_x_moles, olivine_x_moles, T_K, P_bar
    ):

        melt_x_new = melt_x_moles + olivine_x_moles.mul(olivine_amount)
        melt_x_new = melt_x_new.normalise()
        temperature_new = melt_x_new.temperature(
            P_bar=P_bar, offset=self._temperature_offset_parameters, warn=False
        )

        return T_K - temperature_new

    def _root_Kd(
        self, exchange_amount, melt_x_moles, exchange_vector, forsterite, P_bar, kwargs
    ):

        melt_x_new = melt_x_moles + exchange_vector.mul(exchange_amount)
        melt_x_new = melt_x_new.normalise()
        Kd_equilibrium, Kd_real = self.calculate_Kds(
            melt=melt_x_new, P_bar=P_bar, forsterite=forsterite, **kwargs
        )

        return Kd_equilibrium - Kd_real

    def Fe_equilibrate(self, inplace=False, **kwargs):
        """
        Equilibrate Fe and Mg between melt inclusions and their olivine host via diffusive Fe-Mg exchange
        """

        # Get settings
        stepsize = kwargs.get(
            "stepsize", getattr(PEC_configuration, "stepsize_equilibration")
        )
        stepsize = pd.Series(stepsize, index=self.inclusions.index, name="stepsize")
        decrease_factor = getattr(PEC_configuration, "decrease_factor")
        Kd_converge = getattr(PEC_configuration, "Kd_converge")
        dfO2 = configuration.dfO2
        P_bar = self.P_bar

        # Calculate temperature and fO2
        temperatures = self.inclusions.temperature(
            P_bar=P_bar, offset=self._temperature_offset_parameters, warn=False
        )
        fO2 = calculate_fO2(T_K=temperatures, P_bar=P_bar, dfO2=dfO2)
        # Get olivine forsterite contents
        forsterite_host = self._olivine.forsterite

        # def calculate_Fe2FeTotal(mol_fractions, T_K, P_bar, fO2, **kwargs):

        #     Fe3Fe2 = self.calculate_Fe3Fe2(
        #         melt_mol_fractions=mol_fractions,
        #         T_K=T_K,
        #         P_bar=P_bar,
        #         fO2=fO2,
        #         **kwargs,
        #     )

        #     return 1 / (1 + Fe3Fe2)

        # Set up initial data
        mi_moles = self.inclusions.moles()
        Kd_equilibrium, Kd_real = self.calculate_Kds(
            melt=mi_moles,
            T_K=temperatures,
            P_bar=P_bar,
            fO2=fO2,
            forsterite=forsterite_host,
        )
        # Fe-Mg exchange vectors
        FeMg_exchange = pd.DataFrame(0, index=mi_moles.index, columns=mi_moles.columns)
        FeMg_exchange.loc[:, ["FeO", "MgO"]] = [1, -1]
        # Find disequilibrium inclusions
        disequilibrium = pd.Series(
            ~np.isclose(Kd_equilibrium, Kd_real, atol=Kd_converge, rtol=0),
            index=Kd_real.index,
        )
        self._model_results.loc[
            disequilibrium.index, ["isothermal_equilibration", "Kd_equilibration"]
        ] = ~disequilibrium
        # Set stepsizes acoording to Kd disequilibrium
        stepsize.loc[Kd_real < Kd_equilibrium] = -stepsize.loc[
            Kd_real < Kd_equilibrium
        ].copy()
        # Store olivine correction for the current itaration
        olivine_corrected_loop = pd.Series(
            0.0, index=mi_moles.index, name="olivine_corrected", dtype=np.float16
        )

        ##### Main Fe-Mg exhange loop #####
        ###################################
        total_inclusions = mi_moles.shape[0]
        reslice = True

        with alive_bar(
            total=total_inclusions,
            title=f"{'Equilibrating': <13} ...",
        ) as bar:
            # , Pool() as pool
            bar(sum(~disequilibrium) / total_inclusions)
            while sum(disequilibrium) > 0:

                if reslice:
                    # Only reslice al the dataframes and series if the amount of disequilibrium inclusions has changed
                    samples = disequilibrium.index[disequilibrium]
                    stepsize_loop = stepsize.loc[samples]
                    temperatures_loop = temperatures.loc[samples]
                    P_loop = P_bar.loc[samples]
                    FeMg_loop = FeMg_exchange.loc[samples, :]
                    mi_moles_loop = mi_moles.loc[samples, :].copy()
                    fO2_loop = fO2.loc[samples]
                    Fo_host_loop = forsterite_host.loc[samples]
                    Kd_eq_loop = Kd_equilibrium.loc[samples].copy()
                    Kd_real_loop = Kd_real.loc[samples].copy()

                # Exchange Fe and Mg
                mi_moles_loop = mi_moles_loop.add(FeMg_loop.mul(stepsize_loop, axis=0))
                # Calculate new equilibrium Kd and Fe speciation
                Kd_eq_loop, _ = self.calculate_Kds(
                    melt=mi_moles_loop.normalise(),
                    T_K=temperatures_loop,
                    fO2=fO2_loop,
                    P_bar=P_loop,
                    forsterite=Fo_host_loop,
                )
                Fe3Fe2_loop = self.calculate_Fe3Fe2(
                    melt_mol_fractions=mi_moles_loop.normalise(),
                    T_K=temperatures_loop,
                    P_bar=P_loop,
                    fO2=fO2_loop,
                )
                Fe2_FeTotal = self.calculate_Fe2FeTotal(Fe3Fe2=Fe3Fe2_loop)
                melt_FeMg = (mi_moles_loop["FeO"] * Fe2_FeTotal) / mi_moles_loop["MgO"]
                # equilibrium olivine composition in oxide mol fractions
                Fo_equilibrium = 1 / (1 + Kd_eq_loop * melt_FeMg)
                olivine = Olivine(
                    {
                        "MgO": Fo_equilibrium * 2,
                        "FeO": (1 - Fo_equilibrium) * 2,
                        "SiO2": 1,
                    },
                    index=mi_moles_loop.index,
                    columns=mi_moles_loop.columns,
                    units="mol fraction",
                    datatype="oxide",
                )
                olivine = olivine.fillna(1e-6).normalise()

                # list of samples that did not equilibrate
                error_samples = []

                # Single core
                for sample in mi_moles_loop.index:

                    # sample_wtPercent = mi_moles_loop.loc[
                    #     sample
                    # ].convert_moles_wtPercent()
                    temperature_new = mi_moles_loop.loc[sample].temperature(
                        P_bar=P_loop.loc[sample],
                        offset=self._temperature_offset_parameters,
                        warn=False,
                    )
                    # sign = np.sign(temperatures_loop.loc[sample] - temperature_new)

                    min_MgO = -(
                        mi_moles_loop.loc[sample, "MgO"] / olivine.loc[sample, "MgO"]
                    )
                    min_FeO = (
                        -mi_moles_loop.loc[sample, "FeO"] / olivine.loc[sample, "FeO"]
                    )
                    min_val = max(min_MgO, min_FeO) + 1e-3
                    try:
                        olivine_amount = root_scalar(
                            self._root_temperature,
                            args=(
                                mi_moles_loop.loc[sample],
                                olivine.loc[sample],
                                temperatures_loop.loc[sample],
                                P_loop.loc[sample],
                            ),
                            # x0=0,
                            # x1=sign * 0.01,
                            bracket=[min_val, 10],
                        ).root
                        self._model_results.loc[
                            sample, ["isothermal_equilibration", "Kd_equilibration"]
                        ] = True
                    except ValueError:
                        # Catch inclusions that cannot reach isothermal equilibrium.
                        self._olivine_corrected.loc[
                            sample, "equilibration_crystallisation"
                        ] = np.nan
                        error_samples.append(sample)
                        self._model_results.loc[sample, "isothermal_equilibration"] = (
                            False
                        )
                        continue

                    mi_moles_loop.loc[sample] = mi_moles_loop.loc[sample] + olivine.loc[
                        sample
                    ].mul(olivine_amount)
                    # current iteration
                    olivine_corrected_loop.loc[sample] = np.float16(olivine_amount)
                    # Running total
                    self._olivine_corrected.loc[
                        sample, "equilibration_crystallisation"
                    ] += olivine_amount
                # mi_moles_loop = mi_moles_loop.normalise()
                ######################################################################
                # Recalculate Kds
                Kd_eq_loop, Kd_real_loop = self.calculate_Kds(
                    melt=mi_moles_loop.normalise(),
                    T_K=temperatures_loop,
                    P_bar=P_loop,
                    fO2=fO2_loop,
                    forsterite=Fo_host_loop,
                )
                # Find inclusions outside the equilibrium forsterite range
                disequilibrium_loop = pd.Series(
                    ~np.isclose(Kd_eq_loop, Kd_real_loop, atol=Kd_converge, rtol=0),
                    index=Kd_eq_loop.index,
                )
                # Find overcorrected inclusions
                overstepped = ~np.equal(
                    np.sign(Kd_real_loop - Kd_eq_loop), np.sign(stepsize_loop)
                )
                # Reverse one iteration and decrease stepsize for overcorrected inclusions
                decrease_stepsize = np.logical_and(overstepped, disequilibrium_loop)
                if sum(decrease_stepsize) > 0:
                    idx_sz = mi_moles_loop.index[decrease_stepsize]
                    mi_moles_loop.drop(labels=idx_sz, axis=0, inplace=True)
                    Kd_eq_loop.drop(labels=idx_sz, axis=0, inplace=True)
                    Kd_real_loop.drop(labels=idx_sz, axis=0, inplace=True)
                    self._olivine_corrected.loc[
                        idx_sz, "equilibration_crystallisation"
                    ] -= olivine_corrected_loop.loc[idx_sz]
                    stepsize.loc[idx_sz] = stepsize_loop.loc[idx_sz].div(
                        decrease_factor
                    )

                mi_moles.loc[mi_moles_loop.index, :] = mi_moles_loop.values

                Kd_equilibrium[Kd_eq_loop.index] = Kd_eq_loop.copy()
                Kd_real[Kd_real_loop.index] = Kd_real_loop.copy()

                disequilibrium_loop = pd.Series(
                    ~np.isclose(Kd_eq_loop, Kd_real_loop, atol=Kd_converge, rtol=0),
                    index=Kd_eq_loop.index,
                )

                # Remove inclusions that returned equilibration errors from future iterations
                disequilibrium.loc[error_samples] = False

                reslice = not disequilibrium.loc[samples].equals(disequilibrium_loop)
                # reslice = (
                #     sum(~disequilibrium_loop) > 0
                # )  # ~np.array_equal(disequilibrium_new, disequilibrium)
                disequilibrium.loc[disequilibrium_loop.index] = (
                    disequilibrium_loop.values
                )

                # Make sure that disequilibrium stays list-like
                # so that any sliced dataframe remains a dataframe
                # TODO these lines can probably be removed
                try:
                    len(disequilibrium)
                except TypeError:
                    disequilibrium = [disequilibrium]

                bar(sum(~disequilibrium) / total_inclusions)

        corrected_compositions = mi_moles.wt_pc()

        # Set compositions of inclusions with equilibration errors to NaNs.
        idx_errors = self._olivine_corrected.index[
            self._olivine_corrected["equilibration_crystallisation"].isna()
        ]
        corrected_compositions.loc[idx_errors] = np.nan

        # temperatures_new = corrected_compositions.temperature(P_bar=P_bar)
        self.inclusions = corrected_compositions

        if (n := (~self._model_results["isothermal_equilibration"]).sum()) > 0:
            w.warn(f"isothermal equilibration not reached for {n} inclusions")

        if not inplace:
            return (
                corrected_compositions,
                self._olivine_corrected["equilibration_crystallisation"].copy(),
                {"Equilibrium": Kd_equilibrium, "Real": Kd_real},
            )

    def correct_olivine(self, inplace=False, **kwargs):
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

        if not self._model_results["isothermal_equilibration"].any():
            raise RuntimeError("None of the inclusions are equilibrated")

        # remove samples that did not reach isothermal equilibration.
        remove_errors = self._model_results["isothermal_equilibration"]

        # Get settings
        stepsize = kwargs.get(
            "stepsize", getattr(PEC_configuration, "stepsize_crystallisation")
        )
        stepsize = pd.Series(
            stepsize, index=self.inclusions.loc[remove_errors].index, name="stepsize"
        )
        decrease_factor = getattr(PEC_configuration, "decrease_factor")
        FeO_converge = kwargs.get(
            "FeO_converge", getattr(PEC_configuration, "FeO_converge")
        )
        dfO2 = getattr(configuration, "dfO2")
        P_bar = self.P_bar.loc[remove_errors]
        # Inclusion compositions in oxide mol fractions
        mi_moles = self.inclusions.loc[remove_errors].moles()
        # Convert to the total amount of moles after equilibration
        mi_moles = mi_moles.mul(
            (
                1
                + self._olivine_corrected.loc[
                    remove_errors, "equilibration_crystallisation"
                ]
            ),
            axis=0,
        )
        # Olivine forsterite and Mg/Fe
        forsterite = self._olivine.loc[remove_errors].forsterite
        # Fe-Mg exchange vectors
        FeMg_vector = pd.Series(0, index=mi_moles.columns, name="FeMg_exchange")
        FeMg_vector.loc[["FeO", "MgO"]] = [1, -1]
        # Starting FeO and temperature
        FeO = self.inclusions.loc[remove_errors, "FeO"].copy()
        FeO_target = self.FeO_target.copy()
        if self._FeO_as_function:
            FeO_target = self.FeO_function(self.inclusions.loc[remove_errors])
        else:
            FeO_target = self.FeO_target.loc[remove_errors].copy()

        temperature_old = self.inclusions.loc[remove_errors].temperature(
            P_bar=P_bar, offset=self._temperature_offset_parameters, warn=False
        )

        stepsize.loc[FeO > FeO_target] = -stepsize.loc[FeO > FeO_target]
        FeO_mismatch = pd.Series(
            ~np.isclose(FeO, FeO_target, atol=FeO_converge, rtol=0), index=FeO.index
        )
        # self._model_results.loc[FeO_mismatch.index, "FeO_converge"] = ~FeO_mismatch

        ##### OLIVINE MELTING/CRYSTALLISATION LOOP #####
        reslice = True
        total_inclusions = mi_moles.shape[0]
        # list of samples with Kd equilibration or FeO convergence errors
        remove_samples = []
        FeO_converge_error = []
        with alive_bar(
            title=f"{'Correcting':<13} ...",
            total=total_inclusions,
        ) as bar:

            while sum(FeO_mismatch) > 0:

                bar(sum(~FeO_mismatch) / total_inclusions)

                if reslice:
                    samples = FeO_mismatch.index[FeO_mismatch]
                    stepsize_loop = stepsize.loc[samples]
                    mi_moles_loop = mi_moles.loc[samples, :].copy()
                    # idx_olivine = mi_moles_loop.index
                    olivine_loop = self._olivine.loc[samples, :]

                mi_moles_loop = mi_moles_loop + olivine_loop.mul(stepsize_loop, axis=0)

                # mi_moles_loop = mi_moles_loop.normalise()
                self._olivine_corrected.loc[
                    samples, "PE_crystallisation"
                ] += stepsize_loop
                #################################################
                ##### Exchange Fe-Mg to keep Kd equilibrium #####
                # list of samples that did not equilibrate
                Kd_equilibration_error = []

                for sample in mi_moles_loop.index:

                    max_val = (
                        mi_moles_loop.loc[sample, "MgO"] - 1e-3
                    )  # don't let MgO or FeO go to 0
                    min_val = -mi_moles_loop.loc[sample, "FeO"] + 1e-3
                    try:
                        exchange_amount = root_scalar(
                            self._root_Kd,
                            args=(
                                mi_moles_loop.loc[sample],
                                FeMg_vector,
                                forsterite.loc[sample],
                                P_bar.loc[sample],
                                {"dfO2": dfO2},
                            ),
                            # x0=0,
                            # x1=max_val / 10,
                            bracket=[min_val, max_val],
                        ).root
                    except ValueError:
                        # Catch inclusions that cannot reach Kd equilibrium
                        Kd_equilibration_error.append(sample)
                        self._olivine_corrected.loc[sample, "PE_crystallisation"] = (
                            np.nan
                        )
                        exchange_amount = 0.0
                        self._model_results.loc[sample, "Kd_equilibration"] = False

                    mi_moles_loop.loc[sample] = mi_moles_loop.loc[
                        sample
                    ] + FeMg_vector.mul(exchange_amount)

                # mi_moles_loop = mi_moles_loop.normalise()
                #################################################
                # Recalculate FeO
                mi_wtpc_loop = mi_moles_loop.wt_pc()
                FeO = mi_wtpc_loop["FeO"]
                if self._FeO_as_function:
                    FeO_target_loop = self.FeO_function(mi_wtpc_loop)
                    self.FeO_target.loc[mi_wtpc_loop.index] = FeO_target_loop
                else:
                    FeO_target_loop = self.FeO_target.loc[mi_wtpc_loop.index]
                # Find FeO mismatched and overcorrected inclusions
                FeO_mismatch_loop = ~np.isclose(
                    FeO, FeO_target_loop, atol=FeO_converge, rtol=0
                )
                FeO_overstepped = ~np.equal(
                    np.sign(FeO_target_loop - FeO), np.sign(stepsize_loop)
                )
                decrease_stepsize_FeO = np.logical_and(
                    FeO_overstepped, FeO_mismatch_loop
                )

                if sum(decrease_stepsize_FeO) > 0:
                    # Reverse one step and decrease stepsize
                    reverse_FeO = mi_moles_loop.index[decrease_stepsize_FeO]
                    mi_moles_loop.drop(labels=reverse_FeO, axis=0, inplace=True)
                    self._olivine_corrected.loc[
                        reverse_FeO, "PE_crystallisation"
                    ] -= stepsize_loop.loc[reverse_FeO]
                    stepsize.loc[reverse_FeO] = stepsize.loc[reverse_FeO].div(
                        decrease_factor
                    )

                    #################################################
                    # TODO 2024.05.16 CHECK IF THIS CONTINGENCY WORKS
                    #################################################

                    # Stop iterating if stepsize falls below 1e-6
                    FeO_converge_error = list(stepsize.index[abs(stepsize) < 1e-6])
                    # self._olivine_corrected.loc[terminate, "PE_crystallisation"] = 0
                    # mi_moles.loc[terminate] = self.inclusions.moles().loc[
                    #     terminate
                    # ].values
                    self._model_results.loc[FeO_converge_error, "FeO_converge"] = False

                # determine FeO convergence
                mi_wtpc_loop = mi_moles_loop.wt_pc()
                FeO = mi_wtpc_loop["FeO"]
                if self._FeO_as_function:
                    FeO_target = self.FeO_function(mi_wtpc_loop)
                    self.FeO_target.loc[FeO_target.index] = FeO_target
                else:
                    FeO_target = self.FeO_target
                FeO_mismatch_loop = pd.Series(
                    ~np.isclose(FeO, FeO_target, atol=FeO_converge, rtol=0),
                    index=FeO.index,
                )

                # Copy loop data to the main variables
                mi_moles.loc[mi_moles_loop.index, :] = mi_moles_loop.copy()
                # mi_wtPercent = mi_moles.wt_pc()

                # Remove inclusions that returned equilibration or FeO convergence errors from future iterations
                remove_samples = (
                    remove_samples + Kd_equilibration_error + FeO_converge_error
                )

                FeO_mismatch[remove_samples] = False
                self._model_results.loc[Kd_equilibration_error, "Kd_equilibration"] = (
                    False
                )

                reslice = (
                    sum(~FeO_mismatch_loop) > 1
                )  # ~np.array_equal(FeO_mismatch_new, FeO_mismatch)

                reslice = not FeO_mismatch.loc[samples].equals(FeO_mismatch_loop)

                FeO_mismatch.loc[FeO_mismatch_loop.index] = FeO_mismatch_loop.values
                bar(sum(~FeO_mismatch) / total_inclusions)

        corrected_compositions = mi_moles.wt_pc()

        self._model_results.loc[FeO_mismatch.index, "FeO_converge"] = ~FeO_mismatch

        self._olivine_corrected["total_crystallisation"] = self._olivine_corrected[
            ["equilibration_crystallisation", "PE_crystallisation"]
        ].sum(axis=1)
        self.inclusions.loc[corrected_compositions.index] = (
            corrected_compositions.values
        )

        # Set compositions of inclusions with equilibration errors to NaNs.
        idx_errors = self._olivine_corrected.index[
            self._olivine_corrected["PE_crystallisation"].isna()
        ]
        self.inclusions.loc[idx_errors] = np.nan

        if (n := (~self._model_results["Kd_equilibration"]).sum()) > 0:
            w.warn(f"Kd equilibration not reached for {n} inclusions")
        if (n := (~self._model_results["FeO_converge"]).sum()) > 0:
            w.warn(f"FeO not converged for {n} inclusions")

        if inplace:
            return

        return (
            corrected_compositions.copy(),
            self._olivine_corrected.mul(100),
            corrected_compositions.temperature(
                P_bar, offset=self._temperature_offset_parameters, warn=False
            ),
        )

    def correct(self, **kwargs) -> Tuple[Melt, pd.DataFrame, pd.Series]:
        """
        Correct inclusions for PEC
        """

        self.Fe_equilibrate(inplace=True, **kwargs)
        corrected_compositions, olivine_corrected, temperatures = self.correct_olivine(
            inplace=False, **kwargs
        )
        corrected_compositions = pd.concat(
            [corrected_compositions, self._model_results], axis=1
        )

        return corrected_compositions, olivine_corrected, temperatures

    def correct_inclusion(self, index, plot=True, **kwargs) -> pd.DataFrame:
        """Correct a single inclusion for PEC"""

        if type(index) == int:
            index = self.inclusions_uncorrected.index[index]

        inclusion = self.inclusions_uncorrected.loc[index].copy().squeeze()
        olivine = self._olivine.loc[index].copy().squeeze()
        FeO_target = self.FeO_target.loc[index].squeeze()
        P_bar = self.P_bar.loc[index].squeeze()

        if self._FeO_as_function:
            FeO_target = self.FeO_function

        equilibrated, olivine_equilibrated, *_ = equilibration_scalar(
            inclusion, olivine, P_bar, **kwargs
        )
        corrected, olivine_corrected, *_ = crystallisation_correction(
            equilibrated.iloc[-1].copy(),
            olivine,
            FeO_target,
            P_bar,
            eq_crystallised=olivine_equilibrated[-1],
            **kwargs,
        )
        total_corrected = olivine_corrected[-1]

        equilibrated["correction"] = "equilibration"
        corrected["correction"] = "correction"

        total_inclusion = pd.concat([equilibrated, corrected.iloc[1:]], axis=0)

        if plot:
            import matplotlib.lines as l
            import matplotlib.pyplot as plt
            from labellines import labelLine

            set_markers = kwargs.get("markers", True)

            import geoplot as gp

            fontsize = 14

            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=False)
            plt.grid(True, color="whitesmoke")

            colors = gp.colors.flatDesign.by_key()["color"]

            linewidth = 5
            markersize = 90

            FeO_color = tuple(np.repeat(0.25, 3))

            plt.plot(
                equilibrated["MgO"],
                equilibrated["FeO"],
                ["-", ".-"][set_markers],
                color=colors[1],
                # label="equilibration",
                linewidth=linewidth,
                mec="k",
                markersize=10,
                # alpha=0.7,
            )
            plt.plot(
                corrected["MgO"],
                corrected["FeO"],
                ["-", ".-"][set_markers],
                color=colors[2],
                linewidth=linewidth,
                mec="k",
                markersize=10,
                # alpha=0.7,
            )
            ax.scatter(
                equilibrated.loc[equilibrated.index[0], "MgO"],
                equilibrated.loc[equilibrated.index[0], "FeO"],
                marker="^",
                color=colors[1],
                edgecolors="k",
                s=markersize,
                zorder=10,
                label="Glass",
            )
            ax.scatter(
                equilibrated.loc[equilibrated.index[-1], "MgO"],
                equilibrated.loc[equilibrated.index[-1], "FeO"],
                marker="o",
                edgecolors="k",
                color=colors[3],
                s=markersize,
                zorder=10,
                label="Equilibrated",
            )
            ax.scatter(
                corrected.loc[corrected.index[-1], "MgO"],
                corrected.loc[corrected.index[-1], "FeO"],
                marker="s",
                color=colors[2],
                edgecolors="k",
                s=markersize,
                zorder=10,
                label="Corrected",
            )

            middle = sum(ax.get_xlim()) / 2

            if self._FeO_as_function:
                FeO_inital = self.FeO_function(corrected)
                ax.plot(
                    corrected["MgO"],
                    FeO_inital,
                    "-",
                    color=FeO_color,
                )
                FeO_target = sum((min(FeO_inital), max(FeO_inital))) / 2
            else:
                ax.axhline(FeO_target, linestyle="-", color=FeO_color, linewidth=1.5)

            FeO_line = ax.get_lines()[-1]
            try:
                labelLine(
                    FeO_line,
                    x=middle,
                    label="initial FeO",
                    size=fontsize * 0.8,
                    color=FeO_color,
                )
            except ValueError:
                pass

            ax.set_ylim(ax.get_ylim()[0], max((FeO_target * 1.03, ax.get_ylim()[1])))

            ax.set_xlabel("MgO (wt. %)")
            ax.set_ylabel("FeO$^T$\n(wt. %)", rotation=0, labelpad=30)

            handles, labels = ax.get_legend_handles_labels()

            handles = handles + [l.Line2D([0], [0], linewidth=0)]

            labels = labels + [f"{total_corrected:.2f} mol. %\nPEC correction"]

            ax.legend(
                handles,
                labels,
                title=index,
                prop={"family": "monospace", "size": fontsize / 1.5},
                fancybox=False,
                facecolor="white",
            )

        return total_inclusion
