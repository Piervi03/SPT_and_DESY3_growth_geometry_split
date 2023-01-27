from setuptools import setup
from Cython.Build import cythonize

setup(name='multivariate_normal',
      ext_modules=cythonize("multivariate_normal.pyx", annotate=True)
      )
