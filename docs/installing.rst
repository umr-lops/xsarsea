.. _installing:

#############
Installation
#############

Although `xsar`_ is not a required dependancy, `xsar`_ installation is recomended.
All examples in this documentation will use `xsar`_.

Before installing, be sure to activate the xsar conda environement:

.. code-block:: shell

    conda activate xsar

user instalation
................

.. code-block:: shell

    conda install -c conda-forge opencv
    pip install git+https://github.com/umr-lops/xsarsea.git

developement installation
.........................

.. code-block:: shell

    git clone https://github.com/umr-lops/xsarsea.git
    cd xsarsea
    pip install -e .
    pip install -r requirements.txt

Update xsarsea
##############

To update xsar installation, just rerun `pip install`, in your already activated conda environment.

.. code-block:: shell

    pip install -U git+https://github.com/umr-lops/xsarsea.git


.. _xsar: https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsar