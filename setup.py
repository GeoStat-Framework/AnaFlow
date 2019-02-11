# -*- coding: utf-8 -*-
""" AnaFlow - analytical solutions for the groundwater-flow equation"""

import os
import codecs
import re

from setuptools import setup, find_packages


# find __version__ ############################################################


def read(*parts):
    """read file data"""
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    """find version without importing module"""
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


###############################################################################


DOCLINES = __doc__.split("\n")
README = open("README.md").read()
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v3 (GPLv3)
Natural Language :: English
Operating System :: MacOS
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Software Development
Topic :: Utilities
"""

VERSION = find_version("anaflow", "_version.py")

setup(
    name='anaflow',
    version=VERSION,
    maintainer="Sebastian Mueller",
    maintainer_email="sebastian.mueller@ufz.de",
    description=DOCLINES[0],
    long_description=README,
    long_description_content_type="text/markdown",
    author="Sebastian Mueller",
    author_email="sebastian.mueller@ufz.de",
    url='https://github.com/GeoStat-Framework/AnaFlow',
    license='GPL - see LICENSE',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    include_package_data=True,
    install_requires=['numpy >= 1.10.0',
                      'scipy >= 0.19.0'],
    packages=find_packages(exclude=['tests*', 'docs*']),
)
