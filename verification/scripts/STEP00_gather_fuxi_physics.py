import os
import sys
import yaml
from glob import glob
from datetime import datetime

import numpy as np
import xarray as xr

from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

config_name = os.path.realpath('../verif_config.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('verif_ind_start', help='verif_ind_start')
parser.add_argument('verif_ind_end', help='verif_ind_end')
args = vars(parser.parse_args())

verif_ind_start = int(args['verif_ind_start'])
verif_ind_end = int(args['verif_ind_end'])
# ======================= #
model_name = 'fuxi_physics'
filename_prefix = '{}_%Y-%m-%dT%HZ.nc'.format(model_name)
# ======================= #

variables_levels = None
time_intervals = None
base_dir = conf[model_name]['save_loc_rollout']
output_dir = conf[model_name]['save_loc_gather']
# Get list of nc files
all_files_list = vu.get_nc_files(base_dir)

flag_overall = False

while flag_overall is False:
    
    flag_overall = True
    for i in range(verif_ind_start, verif_ind_end):
        # True: process can pass
        flag = vu.process_file_group(all_files_list[i], output_dir, variables_levels, size_thres=1200181629)

        flag_overall = flag_overall and flag
