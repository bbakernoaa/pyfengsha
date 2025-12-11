from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name='pyfengsha',
          version='0.1',
          description="Wrapper around the NOAA ARL FENGSHA dust emission scheme",
          author="Barry D. Baker",
          license='MIT',
          author_email="barry.baker@noaa.gov",
          packages=find_packages(),
          install_requires=['numpy', 'numba'],
          extras_require={
              'xarray': ['xarray', 'dask'],
          }
          )
