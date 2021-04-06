import os
import utils.dataset as dataset
import utils.constants as constants
import utils.skel as skel
import numpy as np

def create_folder(root, run):
    folder = os.path.join(root, run)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def save_args(path, args):
    with open(path, 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write('{} = {}\n'.format(arg, attr))

def save_configs(path, config):
    with open(path, 'w') as f:
        f.write(open(config, 'r').read())
        
def generate_unique_run_name(name, model_save_path, run_save_path):
    run_string = "_run="
    run_count = 0
    not_unique = True
    new_run_name = name + run_string
    while not_unique:
        temp_new_run_name = new_run_name + str(run_count)
        temp_model_save_path = os.path.join(model_save_path, temp_new_run_name)
        temp_run_save_path = os.path.join(run_save_path, temp_new_run_name)
        if os.path.exists(temp_model_save_path) or os.path.exists(temp_run_save_path):
            run_count += 1
        else:
            new_run_name = temp_new_run_name
            not_unique = False
    return new_run_name
    