.. include:: ./substitutions.rst

===============
Model selection
===============

The models available in MagmaPEC can be viewed in the `MagmaPandas documentation <https://magmapandas.readthedocs.io/en/latest/models.html>`_
Please follow the `configuration example <https://magmapec.readthedocs.io/en/latest/notebooks/config.html>`_ for an explanation of how to configure model setup.

MagmaPandas includes a range of models to calculate |fO2|, |kd|, |Fe3Fe2|, olivine liquidus temperatures, and |H2O|-|CO2| saturation pressures. For |Fe3Fe2| and |kd|, calibration datasets are (partially) included, and MagmaPandas has built-in functions to access and plot them. Amongst other things, these functions can be used to compare compositions in your own dataset to those in the calibrations in order to select an appropriate model. The `example notebooks in the MagmaPandas documentation <https://magmapandas.readthedocs.io/en/latest/notebooks/calibration_plots.html#>`_ provide instructions on how to use these functions.

To confirm the accuracy of the models coded in MagmaPandas we performed benchmarks. Results of these benchmarks can be viewed in `the MagmaPandas online documentation <https://magmapandas.readthedocs.io/en/latest/benchmarks.html>`_