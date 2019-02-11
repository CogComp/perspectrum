import os
from setuptools import setup, find_packages

# Utility method to read the README.rst file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

import version
VERSION = version.__version__

CLASSIFIERS = [
    'Development Status :: 3 - Alpha', 'Intended Audience :: Science/Research',
    'Operating System :: Microsoft :: Windows', 'Operating System :: POSIX',
    'Operating System :: Unix', 'Operating System :: MacOS',
    'Programming Language :: Python :: 3.6', 'Topic :: Scientific/Engineering'
]

setup(
    name='perspectrum',
    version=VERSION,
    description=("Discovering Minimal Perspective into Controversial Claims"),
    long_description=read('README.md'),
    url='https://github.com/CogComp/perspectrum',
    author='Cognitive Computation Group',
    author_email='sihaoc@seas.upenn.edu',
    license='Creative Commons (CC-BY-SA)',
    keywords="NLP, natural language processing, information pollution",
    packages=find_packages(exclude=['tests.*', 'tests']),
    install_requires=['configparser', 'django', 'httplib2', 'requests', 'matplotlib', 'PuLP'],
    package_data={'perspectrum': ['config/*.cfg']},
    classifiers=CLASSIFIERS,
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'mock'],
    zip_safe=False)
