################################################
xsarsea: science module for xsar reader
################################################

**xsarsea** is an level-1 SAR product analysis tool.

It works with `xarray`_ datasets, and some of the methods are designed to used `xsar`_ outputs.



Documentation
-------------

Getting Started
...............

:doc:`installing`
~~~~~~~~~~~~~~~~~

Examples
........

Those examples show how to:
    * `Detrend sigma0`_ (also called *roughness* or *nice display*)
    * `Wind streaks direction`_


* :doc:`examples/xsarsea`

* :doc:`examples/streaks`

* :doc:`examples/windspeed_inversion`

* :doc:`examples/windspeed_retrieval_L1`

* :doc:`examples/gmfs_and_luts`

Help & Reference
................

:doc:`basic_api`
~~~~~~~~~~~~~~~~

Get in touch
------------

- Report bugs, suggest features or view the source code on `github`_.

----------------------------------------------

Last documentation build: |today|


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Examples

   examples/xsarsea
   examples/streaks
   examples/windspeed_retrieval_L1
   examples/windspeed_inversion
   examples/gmfs_and_luts

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & Reference

   basic_api

.. _xarray: http://xarray.pydata.org
.. _github: https://github.com/umr-lops/xsarsea
.. _xsar: https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsar
.. _Detrend sigma0: examples/xsarsea.ipynb#Sigma0-detrending
.. _Wind streaks direction: examples/streaks.ipynb#Streaks-analysis

