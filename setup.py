from setuptools import setup, Extension
#import numpy as np

MIxnyn_ext = Extension(
    name="continuousmiestimation._MIxnyn",
    sources=["src/continuousmiestimation/MIxnyn.C"],
    include_dirs=[
        "src/continuousmiestimation",          # so miutils.h is found relative to MIxnyn.C
        #np.get_include(),         # numpy headers, likely needed
    ],
    extra_compile_args=["-fPIC", "-O2", "-shared"],
    #extra_link_args=["-lm"],      # math library for sqrt, log etc.
    language="c++",
)

setup(
    ext_modules=[MIxnyn_ext],
)
