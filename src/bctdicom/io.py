import h5py
import logging

logger = logging.getLogger(__name__)

def read_h5_file(file_path: str, dataset_name: str = 'data'):
    with h5py.File(file_path, "r") as f:
        return f[dataset_name][:]


