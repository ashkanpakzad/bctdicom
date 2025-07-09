import h5py
import numpy as np
import nrrd
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def read_file(file_path: Path):
    if file_path.suffix == ".h5" or file_path.suffix == ".hdf":
        return read_h5(file_path)
    elif file_path.suffix == ".nrrd":
        return read_nrrd(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def write_file(file_path: Path, data: np.ndarray, extra_attrs: dict = None):
    if file_path.suffix == ".h5" or file_path.suffix == ".hdf":
        return write_h5(file_path, data, extra_attrs)
    elif file_path.suffix == ".nrrd":
        return write_nrrd(file_path, data, extra_attrs)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def read_h5(file_path: Path, dataset_name: str = 'data'):
    logger.info(f"Reading H5 file {file_path}/{dataset_name}")
    with h5py.File(file_path, "r") as f:
        data = f[dataset_name][:]
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    return data

def write_h5(file_path: Path, data: np.ndarray, dataset_name: str = 'data', extra_attrs: dict = None):
    logger.info(f"Writing H5 file {file_path}/{dataset_name}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    with h5py.File(file_path, "w") as f:
        f.create_dataset(dataset_name, data=data)
        if extra_attrs is not None:
            logger.info(f"Extra attributes keys to H5: {extra_attrs.keys()}")
            for key, value in extra_attrs.items():
                f.attrs[key] = value

def read_nrrd(file_path: Path, return_header: bool = False):
    logger.info(f"Reading NRRD file {file_path}")
    data, header = nrrd.read(str(file_path))
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    if return_header:
        return data, header
    else:
        return data

def write_nrrd(file_path: Path, data: np.ndarray, extra_attrs: dict = None):
    logger.info(f"Writing NRRD file {file_path}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    if extra_attrs:
        logger.info(f"Extra attributes keys to NRRD header: {extra_attrs.keys()}")
        nrrd.write(str(file_path), data, extra_attrs)
    else:
        nrrd.write(str(file_path), data)