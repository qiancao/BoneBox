# -*- coding: utf-8 -*-

"""
 partially taken from:
     https://github.com/AIM-Harvard/pyradiomics/blob/master/examples/batchprocessing_parallel.py
     
"""

from __future__ import print_function

from collections import OrderedDict
import csv
from datetime import datetime
import logging
from multiprocessing import cpu_count, Pool
import os
import shutil
import threading

import SimpleITK as sitk
     
import radiomics
from radiomics import featureextractor, getFeatureClasses
from radiomics.featureextractor import RadiomicsFeatureExtractor

threading.current_thread().name = 'Main'

# File variables
ROOT = os.getcwd()
PARAMS = os.path.join(ROOT, 'exampleSettings', 'Params.yaml')  # Parameter file
LOG = os.path.join(ROOT, 'log.txt')  # Location of output log file
INPUTCSV = os.path.join(ROOT, 'testCases.csv')
OUTPUTCSV = os.path.join(ROOT, 'results.csv')

# Parallel processing variables
TEMP_DIR = '_TEMP'
REMOVE_TEMP_DIR = True  # Remove temporary directory when results have been successfully stored into 1 file
NUM_OF_WORKERS = cpu_count() - 1  # Number of processors to use, keep one processor free for other work
if NUM_OF_WORKERS < 1:  # in case only one processor is available, ensure that it is used
  NUM_OF_WORKERS = 1
HEADERS = None  # headers of all extracted features

out_dir = "/data/BoneBox-out/"
