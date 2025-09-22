==============================
MagmaPEC quickstart
==============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
   getting_started
   examples
   models
   code_documentation
   /notebooks/benchmark
   support
   license

About
-----
MagmaPEC is a python based tool set for calculating post-entrapment crystallisation extents for olivine-hosted melt inclusions and includes error propagation of various parameters through a Monte-Carlo simulation.

Installation
------------
MagmaPEC can be installed with pip by running:

.. code-block:: bash

    pip install magmapec

in a terminal.
If you want to install from a specific git branch or release, use:

.. code-block:: bash

    pip install git+https://github.com/TDGerve/MagmaPEC.git@tag
    
where *tag* should be repleaced by the release tag or branch name (e.g. *v1.0.0* or *development*)

FeO and Fe2O3 in MagmaPEC
-------------------------

.. important::

    In MagmaPandas, if only FeO or Fe2O3 is included in a composition, it represents total Fe. They only represent :math:`\mathrm{Fe}^{2+}O` and :math:`\mathrm{Fe}^{3+}_{2}\mathrm{O}_{3}`, if both FeO and Fe2O3 are included in the composition.

Indices and tables
------------------


* :ref:`genindex`
* :ref:`search`
