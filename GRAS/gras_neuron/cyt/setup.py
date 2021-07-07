from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension('py_interface', ['cyt_head.pxd', 'cyt_main.pyx', 'pure_cpp.cpp'],
              extra_compile_args=['-std=c++11'],
              language='c++'
              ),
]

setup(ext_modules=cythonize(extensions))
