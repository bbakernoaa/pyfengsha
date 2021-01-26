from numpy.distutils.core import Extension

ext2 = Extension(name='fengsha',
                 sources=['pyfengsha/fengsha.pyf', 'pyfengsha/fengsha.F90'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='pyfengsha',
          version='0.1',
          description="Wrapper around the NOAA ARL FENGSHA dust emission scheme ",
          author="Barry D. Baker",
          lisense='MIT',
          author_email="barry.baker@noaa.gov",
          source=['pyfengsha'],
          packages=['pyfengsha'],
          ext_modules=[ext2],
          install_requires=['numpy']
          )
