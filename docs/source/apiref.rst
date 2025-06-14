API Reference
=============

Factor models (``mosaicperm.factor``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ~mosaicperm.factor.MosaicFactorTest
   ~mosaicperm.factor.MosaicBCV

   :template: 
   ~mosaicperm.factor.ols_residuals

Panel data (``mosaicperm.panel``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ~mosaicperm.panel.MosaicPanelInference
   ~mosaicperm.panel.MosaicPanelTest
   ~mosaicperm.panel.QuadraticMosaicPanelTest

   :template: 
   ~mosaicperm.panel.ols_residuals_panel

Combining results (``mosaicperm.core``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   ~mosaicperm.core.combine_mosaic_tests
   ~mosaicperm.core.combine_mosaic_tests_tseries

Test statistics (``mosaicperm.statistics``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   ~mosaicperm.statistics.mean_abscorr_stat
   ~mosaicperm.statistics.mean_maxcorr_stat
   ~mosaicperm.statistics.quantile_maxcorr_stat
   ~mosaicperm.statistics.mosaic_bcv_stat
   ~mosaicperm.statistics.adaptive_mosaic_bcv_stat
   ~mosaicperm.statistics.approximate_sparse_pcas
   ~mosaicperm.statistics.active_subset

Tilings (``mosaicperm.tilings``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   ~mosaicperm.tilings.Tiling
   ~mosaicperm.tilings.default_factor_tiles
   ~mosaicperm.tilings.random_tiles
   ~mosaicperm.tilings.check_valid_tiling

Other core functions (``mosaicperm.core``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   ~mosaicperm.core.compute_adaptive_pval
   ~mosaicperm.core.MosaicPermutationTest
