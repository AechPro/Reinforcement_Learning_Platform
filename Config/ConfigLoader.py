import json
import os


def load_config(file_path=None, file_name=None):
    default_path = "resources/config_files"
    if file_path is not None:
        if not os.path.exists(file_path):
            print("\nUNABLE TO LOCATE CONFIG FILE IN PATH:\n",file_path,"\n")
            raise FileNotFoundError

    elif file_name is not None:
        file_path = ''.join([default_path,'/',file_name])
        if not os.path.exists(file_path):
            print("\nUNABLE TO LOCATE CONFIG FILE IN PATH:\n",file_path,"\n")
            raise FileNotFoundError

    config = json.load(open(file_path, 'r'))

    return dict(config)
