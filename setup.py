from setuptools import setup, find_packages
import glob

setup(
    name='xsarsea',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    scripts=glob.glob('src/scripts/*.py'),
    url='https://github.com/umr-lops/xsarsea',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        'numpy',  # numba needs numpy<=1.21
        'xarray',
        'opencv-python',
        'importlib-resources',
        'fsspec',
        'aiohttp',
        'numba',
        'scipy',
        'pyyaml',
        'typer',
        'netCDF4',
        'matplotlib',
        'xrft'
    ],
    license='MIT',
    author='Olivier Archer',
    author_email='Olivier.Archer@ifremer.fr',
    description='cmod, detrend, and others sar processing tools over ocean',
    long_description_content_type='text/x-rst',
    long_description ='xsarsea aims at computing geophysical parameters (such as wind, waves or currents) from radar quantities'
)
