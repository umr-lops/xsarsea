from setuptools import setup, find_packages

setup(
    name='xsarsea',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    url='https://github.com/umr-lops/xsarsea',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'numpy',
        'xarray',
        'opencv-python'
    ],
    license='MIT',
    author='Olivier Archer',
    author_email='Olivier.Archer@ifremer.fr',
    description='cmod, detrend, and others sar processing tools over ocean'
)
