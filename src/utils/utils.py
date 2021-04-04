import os

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