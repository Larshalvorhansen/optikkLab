import glob
import importlib

# Get the current directory
current_dir = __file__.rpartition('/')[0]

# Get all Python files in the current directory
file_paths = glob.glob(f"{current_dir}/*.py")

# Import all files and functions
for file_path in file_paths:
    module_name = file_path.rpartition('/')[2].removesuffix('.py')
    module = importlib.import_module(f".{module_name}", __package__)
    globals().update(vars(module))