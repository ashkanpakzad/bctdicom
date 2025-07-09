import h5py
import numpy as np
import nrrd

import logging
logger = logging.getLogger(__name__)

def read_file(file_path: str):
    if file_path.endswith(".h5"):
        return read_h5_file(file_path)
    elif file_path.endswith(".nrrd"):
        return read_nrrd_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def read_h5_file(file_path: str, dataset_name: str = 'data'):
    logger.info(f"Reading H5 file {file_path}/{dataset_name}")
    with h5py.File(file_path, "r") as f:
        data = f[dataset_name][:]
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    return data

def write_h5_file(file_path: str, data: np.ndarray, dataset_name: str = 'data', extra_attrs: dict = {}):
    logger.info(f"Writing H5 file {file_path}/{dataset_name}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    logger.info(f"Extra attributes keys to H5: {extra_attrs.keys()}")
    with h5py.File(file_path, "w") as f:
        f.create_dataset(dataset_name, data=data)
        for key, value in extra_attrs.items():
            f.attrs[key] = value

def read_nrrd_file(file_path: str, return_header: bool = False):
    logger.info(f"Reading NRRD file {file_path}")
    data, header = nrrd.read(file_path)
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    if return_header:
        return data, header
    else:
        return data

def write_nrrd_file(file_path: str, data: np.ndarray, extra_attrs: dict = {}):
    logger.info(f"Writing NRRD file {file_path}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    logger.info(f"Extra attributes keys to NRRD header: {extra_attrs.keys()}")
    nrrd.write(file_path, data, extra_attrs)