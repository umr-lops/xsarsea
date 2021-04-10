.. _installing:

#############
Installation
#############

Although `xsar`_ is not a required dependancy, `xsar`_ installation is recomended.
All examples in this documentation will use `xsar`_.

Before installing, be sure to activate the xsar conda environement:

.. code-block:: shell

    conda activate xsar

user instalation:

.. code-block:: shell

    pip install git+https://gitlab.ifremer.fr/sarlib/xsarsea.git

developement installation:

.. code-block:: shell

    git clone https://gitlab.ifremer.fr/sarlib/xsarsea.git
    cd xsarsea
    pip install -e .
    pip install -r requirements.txt

Update xsar
###########

To update xsar installation, just rerun `pip install`, in yout already activated conda environment.

.. code-block:: shell

    pip install -U git+https://gitlab.ifremer.fr/sarlib/xsarsea.git


.. _xsar: https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsar