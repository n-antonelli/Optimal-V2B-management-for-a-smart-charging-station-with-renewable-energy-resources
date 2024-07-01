import inspect
import os


def get_project_root_dir():
    return f"{get_current_file_path()}"

def get_current_file_path():
    script_path = os.path.abspath(__file__)
    #caller_file_path = os.path.abspath(os.path.dirname(__file__))
    caller_file_path = os.path.abspath(os.path.join(script_path, os.pardir))
    return os.path.dirname(caller_file_path)

def get_files_in_path(path):
    return [f for f in os.listdir(path) \
            if os.path.isfile(os.path.join(path, f))]