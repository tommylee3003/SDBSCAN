import os
from os.path import join

import numpy

# from sklearn._build_utils import get_blas_info
from numpy.distutils.core import setup,Extension
from Cython.Build import cythonize

# cblas_libs, blas_info = get_blas_info()


setup(ext_modules=cythonize(Extension(
    'sdbscan_merge_chain',
    sources=['sdbscan_merge_chain.pyx'],
    language='c++',
    include_dirs=[join('..', 'src', 'cblas'),numpy.get_include()]
)))
