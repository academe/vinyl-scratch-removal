"""
Setup script for building the Cython vinyl scratch removal core library.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Extension modules
extensions = [
    Extension(
        "detection",
        ["detection.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
    ),
    Extension(
        "interpolation",
        ["interpolation.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
    ),
    Extension(
        "vinyl_core",
        ["vinyl_core.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
    ),
]

setup(
    name="vinyl-core",
    version="1.0.0",
    description="Vinyl scratch removal core library (Cython)",
    author="Claude (Anthropic)",
    license="MIT",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'soundfile>=0.10.3',
    ],
    python_requires='>=3.7',
)
