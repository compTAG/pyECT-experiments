from setuptools import setup, Extension
from Cython.Build import cythonize
import os

boost_include = "/opt/homebrew/include"

extensions = [
    Extension(
        "eucalc",
        ["src/eucalc.pyx"],
        include_dirs=["src", boost_include],
        language="c++",
        extra_compile_args=["-std=c++2a"],
    )
]

setup(
    ext_modules=cythonize(extensions)
)
