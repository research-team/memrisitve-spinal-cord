import os
from distutils.core import setup
from os.path import join as pjoin
from Cython.Distutils import build_ext
from distutils.extension import Extension

def find_in_path(name, path):
    """Find a file in a search path"""
    # Adapted fom http://code.activestate.com/recipes/52224
    for directory in path.split(os.pathsep):
        binpath = pjoin(directory, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda(hardpath='/usr/local/cuda/bin/nvcc'):
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        if (nvcc := find_in_path('nvcc', os.environ['PATH'])) is None:
            nvcc = hardpath
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    # form the config: CUDA home and inside folders (lib/include)
    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    # check if paths are exist
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig

def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    compiler = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        compiler(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# find CUDA
CUDA = locate_cuda()

class CustomBuildExt(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

ext = Extension('Interface',
        sources=['Interface.pyx', 'core.cu'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        # we're only going to use certain compiler args with nvcc and not with gcc
        # the implementation of this trick is in customize_compiler() below
        extra_compile_args=dict(gcc=[],
                                nvcc=['-arch=sm_60', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", '-O3']),
        include_dirs=[CUDA['include'], 'src'],
        extra_link_args=["-std=c++11"])

setup(name='gpu_gras',
      author='ncnlab',
      version='0.1',
      ext_modules=[ext],
      cmdclass={'build_ext': CustomBuildExt},
      # Since the package has c code, the egg cannot be zipped
      zip_safe=False)