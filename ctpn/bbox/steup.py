from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


numpy_include = np.get_include()
setup(ext_modules=cythonize(Extension(name="bbox", sources=["bbox.pyx"])), include_dirs=[numpy_include])
setup(ext_modules=cythonize(Extension(name="nms", sources=["nms.pyx"])), include_dirs=[numpy_include])
