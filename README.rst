xsarsea
#######

.. image:: https://img.shields.io/pypi/v/xsarsea.svg
        :target: https://pypi.python.org/pypi/xsarsea

.. image:: https://readthedocs.org/projects/xsarsea/badge/?version=latest
        :target: https://xsarsea.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

CMOD, sigma0 detrend, and others sar processing tools over ocean.

Installation
------------

`xsarsea` is supposed to use  [xsar](https://github.com/umr-lops/xsar) objects to map SAR products into `xarray` Datasets. 

now you can install xsarsea

.. code-block:: bash

    conda activate xsar

for user:
_________

.. code-block:: bash

    pip install git+https://github.com/umr-lops/xsarsea.git


for development:
________________

.. code-block:: bash

    pip install git+https://github.com/umr-lops/xsarsea.git
    cd xsarsea
    pip install -r requirements.txt
    pip install -e .


Documentation and examples
--------------------------

https://cerweb.ifremer.fr/datarmor/doc_sphinx/xsarsea/