import numpy as np
from scipy import ndimage
from tqdm import track

import logging
logger = logging.getLogger(__name__)

def thick_axial(input: np.ndarray, input_mm: float = [1.0, 1.0, 1.0], axis: int = 0, 
                thickness_mm: float = 3, spacing_mm: float = 1.5, threshold: float = 2e-10, 
                MIP: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute thick slices from input array along specified axis.

    Args:
        input (np.ndarray): Input array to compute thick slices from
        input_mm (float, optional): Spacing between voxels in mm. Defaults to [1.0,1.0,1.0].
        axis (int, optional): Axis to compute thick slices along (0,1,2). Defaults to 0.
        thickness_mm (float, optional): Thickness of slices in mm. Defaults to 3.
        spacing_mm (float, optional): Spacing between thick slices in mm. Defaults to 1.5.
        threshold (float, optional): Threshold for high intensity voxels. Defaults to 2e-10.
        MIP (bool, optional): Use maximum intensity projection instead of mean. Defaults to False.

    Returns:
        np.ndarray: Array containing thick slices
        np.ndarray: Spacing between thick slices in mm
        float: Thickness of thick slices in mm
    """
    logger.debug(f"Input array shape: {input.shape}")
    logger.debug(f"Input spacing (mm): {input_mm}")
    logger.debug(f"Input axis: {axis}")
    logger.debug(f"Desired slice thickness (mm): {thickness_mm:.1f}")
    logger.debug(f"Desired slice spacing (mm): {spacing_mm:.1f}")

    # move chosen axis to first dimension
    source_array = input.copy()
    source_array = np.moveaxis(source_array, axis, 0)
    output_mm = input_mm.copy()

    # get original dimension spacing
    nThin = source_array.shape[0]
    thin_spacing_mm = input_mm[axis]
    spacing = int(spacing_mm / thin_spacing_mm)
    thickness = int(thickness_mm / thin_spacing_mm)

    # convert to ints and compute output thickness and spacing in mm
    output_spacing_mm = thin_spacing_mm * spacing
    output_thickness_mm = thin_spacing_mm * thickness
    output_mm[axis] = output_spacing_mm

    # compute number of thick slices
    nThick = 1 + int((nThin - thickness) / spacing)
    output_array = np.zeros([nThick] + list(source_array.shape[1:]), dtype=source_array.dtype)

    # warn if thickness or spacing is not integer multiples of thin slice spacing
    if not np.isclose(thickness_mm, round(output_thickness_mm)):
        logger.warning(f"Desired thickness {thickness_mm:.1f}mm is not an integer multiple of source spacing {thin_spacing_mm:.1f}mm")
        logger.warning(f"Using {output_thickness_mm:.1f}mm instead")    
    if not np.isclose(spacing_mm, round(output_spacing_mm)):
        logger.warning(f"Desired spacing {spacing_mm:.1f}mm is not an integer multiple of source spacing {thin_spacing_mm:.1f}mm")
        logger.warning(f"Using {output_spacing_mm:.1f}mm instead")
        
    logger.debug(f"Slice thickness (pixels): {thickness}")
    logger.debug(f"Slice spacing (pixels): {spacing}")
    logger.debug(f"Actual slice thickness (mm): {output_thickness_mm:.1f}")
    logger.debug(f"Actual slice spacing (mm): {output_spacing_mm:.1f}")

    logger.info(f"Computing {nThin} thin slices into {nThick} thick slices")
    # identify high voxels
    if threshold is not None:
        logger.debug("Processing high intensity voxels")
        high = source_array > threshold
        logger.debug(f"Found {np.sum(high)} voxels above threshold")
        
        # count neighbours, we exclude isolated upper voxels as noise
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0  

        # convolve to count neighbours
        n_neighbours = ndimage.convolve(high.astype(np.uint8), kernel)
        logger.debug(f"Neighbor count range: {n_neighbours.min()} to {n_neighbours.max()}")

        # remove isolated voxels
        high[n_neighbours<=1] = 0
        logger.debug(f"Removed {np.sum(n_neighbours<=1)} isolated voxels")

    # init loop
    for i in track(range(nThick),total=nThick,description='Thick slicing: '):
        i_min=i*spacing
        i_max=i_min+thickness
        
        # compute thick slice by mean
        thins = source_array[i_min:i_max, ...]
        if MIP:
            # Maximum Intensity Projection
            thick = np.max(thins, axis=0)
        else:
            # Use average
            thick = np.mean(thins, axis=0)

        if threshold is not None:
            # get high voxels 
            thins_upper = high[i_min:i_max, ...]
            # identify voxels above threshold in thick
            thick_upper = np.any(thins_upper, axis=0)
            
            # get mean of voxels above threshold
            thins_hi = thins.copy()
            thins_hi[~thins_upper] = np.nan
            thick_hi = np.nanmean(thins_hi, axis=0)
            # set voxels above threshold to mean above threshold
            thick[thick_upper] = thick_hi[thick_upper]
            
        output_array[i, ...] = thick
        
    logger.debug("Moving output array back to original dimensions")
    output_array = np.moveaxis(output_array, 0, axis)
    logger.info(f"Thick slices shape: {output_array.shape}")

    return output_array, output_mm, output_thickness_mm