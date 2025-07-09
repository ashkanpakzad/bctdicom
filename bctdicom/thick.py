import numpy as np
from scipy import ndimage
from tqdm import tqdm
import warnings

import logging
logger = logging.getLogger(__name__)

def thick_reslice_axis(input: np.ndarray, input_mm: list = None, axis: int = 0, 
                thickness_mm: float = 3, spacing_mm: float = 1.5, threshold: float = None, 
                MIP: bool = False) -> tuple[np.ndarray, list, float]:
    """Compute thick slices from input array along specified axis.

    Args:
        input (np.ndarray): Input array to compute thick slices from
        input_mm (list, optional): Spacing between voxels in mm. Defaults to [1.0,1.0,1.0].
        axis (int, optional): Axis to compute thick slices along (0,1,2). Defaults to 0.
        thickness_mm (float, optional): Thickness of slices in mm. Defaults to 3.
        spacing_mm (float, optional): Spacing between thick slices in mm. Defaults to 1.5.
        threshold (float, optional): Threshold for upper intensity voxels. Defaults to None.
        MIP (bool, optional): Use maximum intensity projection instead of mean. Defaults to False.

    Returns:
        np.ndarray: Array containing thick slices
        list: Spacing between thick slices in mm
        float: Thickness of thick slices in mm
    """
    if input_mm is None:
        input_mm = [1.0, 1.0, 1.0]
        
    logger.debug(f"Input array shape: {input.shape}")
    logger.debug(f"Input spacing (mm): {input_mm}")
    logger.debug(f"Input axis: {axis}")
    logger.debug(f"Desired slice thickness (mm): {thickness_mm:.1f}")
    logger.debug(f"Desired slice spacing (mm): {spacing_mm:.1f}")

    # Prepare arrays
    source_array = input.copy()
    source_array = np.moveaxis(source_array, axis, 0)
    
    # Calculate parameters
    nThin, spacing, thickness, output_spacing_mm, output_thickness_mm, output_mm = _calculate_parameters(
        source_array, input_mm, axis, thickness_mm, spacing_mm
    )
    
    # Create output array
    nThick = int(1 + (nThin - thickness) / spacing)
    output_array = _create_output_array(nThick, source_array)
    
    # Validate parameters
    thin_spacing_mm = input_mm[axis]
    _validate_parameters(thickness_mm, spacing_mm, output_thickness_mm, output_spacing_mm, thin_spacing_mm)
    
    logger.debug(f"Slice thickness (pixels): {thickness}")
    logger.debug(f"Slice spacing (pixels): {spacing}")
    logger.debug(f"Actual slice thickness (mm): {output_thickness_mm:.1f}")
    logger.debug(f"Actual slice spacing (mm): {output_spacing_mm:.1f}")

    # Process upper intensity voxels if threshold is set
    upper = _process_upper_intensity_voxels(source_array, threshold)
    
    # Compute all slices
    _compute_all_slices(source_array, nThick, spacing, thickness, output_array, MIP, threshold, upper)
    
    # Finalize output
    logger.debug("Moving output array back to original dimensions")
    output_array = np.moveaxis(output_array, 0, axis)
    logger.info(f"Thick slices shape: {output_array.shape}")

    return output_array, output_mm, output_thickness_mm

def _calculate_parameters(source_array: np.ndarray, input_mm: list, axis: int, 
                        thickness_mm: float, spacing_mm: float) -> tuple[int, int, int, float, float, list]:
    """Calculate spacing, thickness, and output parameters."""
    nThin = source_array.shape[0]
    thin_spacing_mm = input_mm[axis]
    spacing = int(spacing_mm / thin_spacing_mm)
    thickness = int(thickness_mm / thin_spacing_mm)
    
    output_spacing_mm = thin_spacing_mm * spacing
    output_thickness_mm = thin_spacing_mm * thickness
    output_mm = input_mm.copy()
    output_mm[axis] = output_spacing_mm
    return nThin, spacing, thickness, output_spacing_mm, output_thickness_mm, output_mm


def _validate_parameters(thickness_mm: float, spacing_mm: float, 
                       output_thickness_mm: float, output_spacing_mm: float, 
                       thin_spacing_mm: float) -> None:
    """Validate parameters and log warnings for non-integer multiples."""
    if not np.isclose(thickness_mm, output_thickness_mm):
        logger.warning(f"Desired thickness {thickness_mm:.1f}mm is not an integer multiple of source spacing {thin_spacing_mm:.1f}mm")
        logger.warning(f"Using {output_thickness_mm:.1f}mm instead")    
    if not np.isclose(spacing_mm, output_spacing_mm):
        logger.warning(f"Desired spacing {spacing_mm:.1f}mm is not an integer multiple of source spacing {thin_spacing_mm:.1f}mm")
        logger.warning(f"Using {output_spacing_mm:.1f}mm instead")


def _create_output_array(nThick: int, source_array: np.ndarray) -> np.ndarray:
    """Create the output array with correct shape and dtype."""
    return np.zeros([nThick] + list(source_array.shape[1:]), dtype=source_array.dtype)


def _process_upper_intensity_voxels(source_array: np.ndarray, threshold: float) -> np.ndarray:
    """Process upper intensity voxels and remove isolated ones."""
    if threshold is None:
        return None
        
    logger.debug("Processing upper intensity voxels")
    upper = source_array > threshold
    logger.debug(f"Found {np.sum(upper)} voxels above threshold")
    
    # count neighbours, we exclude isolated upper voxels as noise
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0  

    # convolve to count neighbours
    n_neighbours = ndimage.convolve(upper.astype(np.uint8), kernel)
    logger.debug(f"Neighbor count range: {n_neighbours.min()} to {n_neighbours.max()}")

    # remove isolated voxels
    upper[n_neighbours<=1] = 0
    logger.debug(f"Removed {np.sum(n_neighbours<=1)} isolated voxels")
    
    return upper


def _compute_thick_slice(thins: np.ndarray, MIP: bool, 
                        threshold: float = None, thins_upper: np.ndarray = None) -> np.ndarray:
    """Compute a single thick slice from thin slices."""
    if MIP:
        # Maximum Intensity Projection
        thick = np.max(thins, axis=0)
    else:
        # Use average
        thick = np.mean(thins, axis=0)

    if threshold is not None and thins_upper is not None:
        # identify voxels above threshold in thick
        thick_upper = np.any(thins_upper, axis=0)
        
        # get mean of voxels above threshold
        thins_upper_values = thins.copy()
        thins_upper_values[~thins_upper] = np.nan
        with warnings.catch_warnings():
            # ignore warnings about expected empty slices
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            thick_upper_mean = np.nanmean(thins_upper_values, axis=0)
        # set voxels above threshold to mean above threshold
        thick[thick_upper] = thick_upper_mean[thick_upper]
        
    return thick


def _compute_all_slices(source_array: np.ndarray, nThick: int, spacing: int, thickness: int,
                       output_array: np.ndarray, MIP: bool, threshold: float = None, 
                       upper: np.ndarray = None) -> np.ndarray:
    """Compute all thick slices."""
    logger.info(f"Computing {source_array.shape[0]} thin slices into {nThick} thick slices")
    
    for i in tqdm(range(nThick), total=nThick, desc='Thick slicing: '):
        i_min = i * spacing
        i_max = i_min + thickness
        
        # compute thick slice
        thins = source_array[i_min:i_max, ...]
        
        if threshold is not None:
            thins_upper = upper[i_min:i_max, ...]
            thick = _compute_thick_slice(thins, MIP, threshold, thins_upper)
        else:
            thick = _compute_thick_slice(thins, MIP)
            
        output_array[i, ...] = thick
    
    return output_array


