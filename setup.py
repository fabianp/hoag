#! /usr/bin/env python
#
# Copyright (C) 2012 Mathieu Blondel

import sys
import os
from setuptools import setup


DISTNAME = 'hoag'
DESCRIPTION = "XXX"
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Fabian Pedregosa'
MAINTAINER_EMAIL = 'f@bianp.net'
URL = 'https://github.com/fabianp/hoag'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/fabianp/hoag'
VERSION = '0.1.0'


if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          packages=['hoag'],
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
             ]
          )
