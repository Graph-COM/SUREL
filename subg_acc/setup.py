from distutils.core import setup, Extension
import numpy

module1 = Extension('surel_gacc',
                    sources = ['graph_acc.c'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-lgomp'],
                    include_dirs=[numpy.get_include()])

setup (name = 'SUREL_GAcc',
       version = '1.1',
       description = 'This is a package for accelerated graph operations in SUREL framework.',
       ext_modules = [module1])
