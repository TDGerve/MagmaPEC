from functools import partial
from multiprocessing import Pool, cpu_count

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

from IPython.display import clear_output
from MagmaPandas.MagmaFrames import Melt, Olivine

from MagmaPEC.error_propagation.FeOi_error_propagation import FeOi_prediction
from MagmaPEC.error_propagation.MC_parameters import PEC_MC_parameters
from MagmaPEC.PEC_model import PEC

_n_cores_free = 2  # use all but 2 cpu cores during multithread operations
_n_cores = max(1, cpu_count() - _n_cores_free)


class PEC_MC:
    """
    Class for post-entrapment crystallisation (PEC) correction of olivine-hosted melt inclusions, with model and input parameter errors propagated in a Monte Carlo (MC) simulation.

    Parameters
    ----------
    inclusions : :py:class:`~magmapandas:MagmaPandas.MagmaFrames.melt.Melt`
        melt inclusion compositions in oxide wt. %.
    olivines : :py:class:`~magmapandas:MagmaPandas.MagmaFrames.olivine.Olivine`
        olivine compositions in oxide wt. %.
    P_bar : float, pandas Series
        pressures in bar
    FeO_target: float, pandas Series, :py:class:`~MagmaPEC.error_propagation.FeOi_prediction`
        melt inclusion initial FeO content, as fixed value (float, Series) or predictive equation based on melt composition (FeOi_prediction)
    MC_parameters : :py:class:`~MagmaPEC.error_propagation.PEC_MC_parameters`
        model and input parameter errors.

    Attributes
    ----------
    pec : pandas DataFrame
        average PEC extents (%) of the MC model, with errors as one standard deviation.
    inclusions_corr : pandas DataFrame
        averages of corrected melt inclusion compositions of the MC model.
    inclusions_stddev : pandas DataFrame
        errors on ``inclusions_corr`` as one standard deviation.
    inclusions_MC : Dict[str, pd.DataFrame]
        corrected melt compositions per inclusion for all MC iterations.
    pec_MC : pandas DataFrame
        modelled PEC extents for all MC interations.
    """

    def __init__(
        self,
        inclusions: Melt,
        olivines: Olivine,
        P_bar: float | pd.Series,
        FeO_target: float | FeOi_prediction,
        MC_parameters: PEC_MC_parameters,
    ):

        self.inclusions = inclusions
        self.olivines = olivines
        self.P_bar = P_bar
        self.FeO_target = FeO_target

        self.parameters = MC_parameters

        self.MC_inputs = np.array([])

    def run(self, n: int, multicore=True):
        """
        Run the PEC correction Monte Carlo loop.

        Parameters
        ----------
        n : int
            number of iterations in the Monte Carlo loop
        multicore : boolean
            run calculations in parallel across multiple cpu cores
        """
        n_cores = _n_cores if multicore else 1

        self._run_multicore(n=n, n_cores=n_cores)

    def _run_multicore(self, n: int, n_cores: int):
        """
        Run the PEC correction Monte Carlo loop on multiple cpu cores.

        Parameters
        ----------
        n : int
            number of iterations in the Monte Carlo loop
        n_cores: int
            number
        """
        self.pec_MC = pd.DataFrame(
            columns=self.inclusions.index, index=pd.Series(range(n), name="iteration")
        )
        self.inclusions_MC = {
            name: Melt(
                columns=list(self.inclusions.columns)
                + ["isothermal_equilibration", "Kd_equilibration", "FeO_converge"],
                index=pd.Series(range(n), name="iteration"),
            )
            for name in self.inclusions.index
        }

        self.parameters.get_parameters(n=n)

        parameters = [
            [i, *v] for i, v in enumerate(zip(*self.parameters._get_iterators()))
        ]

        inputs_idx = np.array([])

        with (
            Pool(processes=n_cores) as pool,
            alive_bar(
                n,
                spinner=None,
                length=25,
                manual=True,
                theme="smooth",
                force_tty=True,
                title="Monte Carlo iterations...",
                title_length=25,
            ) as bar,
        ):

            results = pool.imap_unordered(self._PEC_multicore, parameters)
            pool.close()

            finished = 0
            bar(0 / n)

            for i, (melts_corr, pec, checks, inputs) in results:
                finished += 1
                bar(finished / n)

                self.MC_inputs = np.append(self.MC_inputs, inputs)
                inputs_idx = np.append(inputs_idx, i)

                for name, row in melts_corr.iterrows():
                    self.inclusions_MC[name].loc[i] = pd.concat([row, checks.loc[name]])
                    self.pec_MC.loc[i, name] = pec.loc[name, "total_crystallisation"]

        self.MC_inputs = self.MC_inputs[np.argsort(inputs_idx)]
        self._calculate_errors()

    def _PEC_multicore(self, parameters):
        """
        Wrapper for multiprocess calling of PEC corrections.
        """

        i = parameters[0]

        melt_MC, olivine_MC, pressure_MC, FeOi = self._process_MC_params(
            *parameters[1:5]
        )

        inputs = {
            "inclusions": melt_MC,
            "olivines": olivine_MC,
            "P_bar": pressure_MC,
            "FeO_target": FeOi,
            "Fe3Fe2_offset_parameters": parameters[5],
            "Kd_offset_parameters": parameters[6],
            "temperature_offset_parameters": parameters[7],
        }

        model = PEC(**inputs)

        melts_corr, pec, checks = model.correct(progressbar=False)

        return i, (melts_corr, pec, checks, inputs)

    def _process_MC_params(self, melt_err, olivine_err, pressure_err, FeOi_err):

        # melt_err = melt_err[1] if isinstance(melt_err, tuple) else melt_err
        melt_MC = self.inclusions[self.inclusions.elements].add(melt_err, axis=1)
        melt_MC[melt_MC < 0] = 0.0
        # olivine_err = olivine_err[1] if isinstance(olivine_err, tuple) else olivine_err
        olivine_MC = self.olivines[self.olivines.elements].add(olivine_err, axis=1)
        olivine_MC[olivine_MC < 0] = 0.0

        pressure_MC = self.P_bar.add(pressure_err)
        pressure_MC[pressure_MC < 0] = 1

        if isinstance(FeOi_err, (float, int)):
            # for a fixed FeO error
            if isinstance(self.FeO_target, (float, int, pd.Series, np.ndarray)):
                FeOi = self.FeO_target + FeOi_err
            elif isinstance(self.FeO_target, FeOi_prediction):
                self.FeO_target._intercept += FeOi_err
                FeOi = partial(
                    self.FeO_target._FeO_initial_func,
                    coefficients=self.FeO_target.coefficients,
                )

        elif isinstance(FeOi_err, (pd.Series, np.ndarray)):
            # for errors per inclusion
            if isinstance(self.FeO_target, FeOi_prediction):
                raise TypeError(
                    "array-like FeOi_error is not supported with FeOi_prediction!"
                )
            elif isinstance(self.FeO_target, (float, int, pd.Series, np.ndarray)):
                FeOi = self.FeO_target + FeOi_err

        elif isinstance(FeOi_err, tuple):
            # for errors on linear regression coefficients
            if "intercept" in FeOi_err[1].index:
                FeOi = partial(
                    self.FeO_target._FeO_initial_func, coefficients=FeOi_err[1]
                )
            elif FeOi_err[1].index.equals(self.inclusions.index):
                # for FeO errors per inclusion
                FeOi = self.FeO_target + FeOi_err[1]

        else:
            raise TypeError(
                "FeOi_error filetype not supported, please use float, int, or array-like (numpy, pandas)"
            )

        return melt_MC, olivine_MC, pressure_MC, FeOi

    def _calculate_errors(self):

        self.pec = pd.concat([self.pec_MC.mean(), self.pec_MC.std()], axis=1).astype(
            float
        )
        self.pec.columns = ("pec", "stddev")

        self.inclusions_corr = Melt(
            index=self.inclusions.index,
            columns=self.inclusions.elements,
            units="wt. %",
            datatype="oxide",
            dtype=float,
        )
        colnames = [f"{n}_stddev" for n in self.inclusions.elements]
        self.inclusions_stddev = pd.DataFrame(
            index=self.inclusions.index, columns=colnames, dtype=float
        )

        for inclusion, df in self.inclusions_MC.items():
            self.inclusions_corr.loc[inclusion] = df[df.elements].mean().values
            self.inclusions_stddev.loc[inclusion] = df[df.elements].std().values

        try:
            self.inclusions_stddev.drop(columns=["total_stddev"], inplace=True)
        except KeyError:
            pass
