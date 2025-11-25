import typer
from pathlib import Path
from bctdicom.thick import thick_reslice_axis
from bctdicom.util import read_file, write_file
from bctdicom.dicom import ScaleTo12Bit, write_vol_dicom
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def thick_reslice(input_file: Path, 
                  output_file: Path,
                  axis: int = 1,
                  mm: tuple[float, float, float] = (0.1, 0.1, 0.1), 
                  thickness_mm: float = 3,
                  spacing_mm: float = 1.5,
                  threshold: float = 2e-10,
                  MIP: bool = False,
                  verbose: bool = False):
    '''
    Reslice a BCT beta-scale image in float32 NRRD/HDF5 format to a thick-sliced 
    image in float32 NRRD/HDF5 format.

    Mean thick-slices are taken along the specified axis. 
    Thickness and spacing must be divisible by the voxel spacing.
    
    Args:
        input_file (Path): Input NRRD file path
        output_file (Path): Output NRRD file path
        axis (int): Axis along which to slice the data
        mm (list[float]): Voxel spacing in mm [x, y, z]
        thickness_mm (float): Slice thickness in mm
        spacing_mm (float): Slice spacing in mm
        threshold (float): Threshold for calcifications
        MIP (bool): Use maximum intensity projection (MIP)
        verbose (bool): Enable verbose logging
    '''
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.info(f"Thick slicing {input_file} to {output_file}")
    logger.debug(f"mm: {mm}")

    data = read_file(input_file)
    output_file = Path(output_file)
    mm = list[float](mm)
    thick_data, mm_out, thickness_mm = thick_reslice_axis(data, mm, axis, thickness_mm, spacing_mm, threshold, MIP)
    header = {'units': ['mm', 'mm', 'mm'], 'spacings': mm_out, 'thickness': float(thickness_mm)}
    write_file(output_file, thick_data, extra_attrs=header)

@app.command()
def thick_reslicing_dcm(input_file: Path, 
                        output_dir: Path,
                        mm: tuple[float, float, float] = (0.1, 1.5, 0.1), 
                        axis: int = 1,
                        thickness_mm: float = 3.0,
                        file_prefix: str = 'ax',
                        dcm_name: str = 'name',
                        dcm_sex: str = 'F',
                        dcm_laterality: str = 'R',
                        dcm_details: str = 'details',
                        flip_axes: list[int] = [2],
                        swap_axes: list[int] = [0, 2],
                        verbose: bool = False):
    '''
    Convert a thick-sliced BCT beta-scale image in float32 NRRD/HDF5 format 
    to DICOM uint16 format. 
    
    Adds necessary metadata for radiology reporting.

    Args:
        input_file (Path): Input NRRD file path
        output_dir (Path): Output directory for DICOM files
        mm (list[float]): Voxel spacing in mm [x, y, z]
        axis (int): Axis along which to slice the data
        thickness_mm (float): Slice thickness in mm
        file_prefix (str): Prefix for output DICOM files
        dcm_name (str): Patient name/ID
        dcm_sex (str): Patient sex (M/F/O)
        dcm_laterality (str): Image laterality (L/R)
        dcm_details (str): Study/series details
        flip_axes (list[int]): List of axes to flip (e.g., [1, 2] to flip axes 1 and 2)
        swap_axes (list[int]): List of axes to swap (e.g., [0, 2] to swap axes 0 and 2)
        verbose (bool): Enable verbose logging
    '''
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.info(f"Converting {input_file} to DICOM {output_dir}")
    mm = list[float](mm)

    input_file = Path(input_file)
    output_dir = Path(output_dir)
    # TODO read in header spacing and thickness
    data = read_file(input_file)

    # Apply custom axis flips if specified
    if flip_axes is not None:
        data = np.flip(data, axis=flip_axes)
        logger.debug(f"Flipped axis {flip_axes}")

    if swap_axes is not None:
        data = np.swapaxes(data, swap_axes[0], swap_axes[1])
        logger.debug(f"Swapped axes {swap_axes}")

    # scale 32 keV beta image to 12-bit
    scaler = ScaleTo12Bit(in_min=5.0e-11, in_max=7.0e-10, out_max=4095)
    data = scaler(data)

    # write DICOM
    write_vol_dicom(output_dir, data, axis=axis, mm=mm, thickness_mm=thickness_mm, file_prefix=file_prefix, dcm_name=dcm_name, dcm_sex=dcm_sex, dcm_laterality=dcm_laterality, dcm_details=dcm_details)

@app.command()
def nrrd_to_h5(input_file: Path, output_file: Path, verbose: bool = False):
    '''
    Convert a NRRD file to a HDF5 file.
    '''
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.info(f"Converting {input_file} to {output_file}")
    data = read_file(input_file)
    write_file(output_file, data)
    
if __name__ == "__main__":
    app()