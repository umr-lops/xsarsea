################################################
xsarsea: science module for xsar reader
################################################

**xsarsea** is an level-1 SAR product analysis tool.

It works with `xarray`_ datasets, and might be used with `xsar`_.



Documentation
-------------

Getting Started
...............

* :doc:`installing`
* :doc:`examples`

Examples
........

Followings examples show how to:
    * `Detrend sigma0`_ (also called *roughness* or *nice display*)
    * `Wind streaks direction`_ (⚠️work in progress⚠)


* :doc:`examples/detrend`

Help & Reference
................

* :doc:`basic_api`

Get in touch
------------

- Report bugs, suggest features or view the source code on `gitLab`_.

----------------------------------------------

Last documentation build: |today|


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing
   examples

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Examples

   examples/detrend

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & Reference

   basic_api

.. _xarray: http://xarray.pydata.org
.. _gitLab: https://gitlab.ifremer.fr/sarlib/xsarsea
.. _xsar: https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsar
.. _Detrend sigma0: examples/detrend.ipynb#Sigma0-detrending
.. _Wind streaks direction: examples/detrend.ipynb#Wind-streaks-direction