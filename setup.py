from setuptools import setup, find_packages

setup(
    name='xsarsea',
    package_dir={'': 'src'},
    packages=find_packages(),
    url='https://gitlab.ifremer.fr/sarlib/xsarsea',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[],
    license='GPL',
    author='Olivier Archer',
    author_email='Olivier.Archer@ifremer.fr',
    description='cmod, detrend, and others sar processing tools over ocean'
)
