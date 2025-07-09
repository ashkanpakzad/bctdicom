import pydicom
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def write_vol_dicom(path: Path | str, 
                    data: np.ndarray, 
                    mm: list[float] = [1.0, 1.0, 1.0],
                    axis: int = -1, 
                    thickness_mm: float = 1.0,
                    file_prefix: str = 'ax', 
                    dcm_name: str = 'name', 
                    dcm_sex: str = 'F', 
                    dcm_laterality: str = 'R', 
                    dcm_details: str = 'details') -> None:
    """Write NumPy array as DICOM files.
    
    Args:
        path (Path | str): Output directory path
        data (np.ndarray): Input data array (12-bit range 0-4095)
        file_prefix (str): Prefix for output files
        axis (int): Axis along which to slice the data
        mm (list[float]): Voxel spacing in mm [x, y, z]
        thickness_mm (float): Slice thickness in mm
        dcm_name (str): Patient name/ID
        dcm_sex (str): Patient sex (M/F/O)
        dcm_laterality (str): Image laterality (L/R)
        dcm_details (str): Study/series details
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Writing DICOM files to {path}")
    
    data = _prepare_data(data)
    
    nslices = data.shape[axis]
    ndigits = len(str(nslices))
    
    base_ds = _create_base_dataset(dcm_name, dcm_sex, dcm_laterality, 
                                   dcm_details, mm, axis, thickness_mm)

    logger.info(f"Writing {nslices} DICOM slices")
    for i in tqdm(range(nslices)):
        slice_data = np.take(data, indices=i, axis=axis)
        _write_slice(slice_data, i, path, file_prefix, ndigits, base_ds)
    
    logger.info(f"Successfully wrote {nslices} DICOM files to {path}")

def _prepare_data(data: np.ndarray) -> np.ndarray:
    """Prepare data for DICOM writing."""
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    # Ensure uint16 and little endian
    data = data.squeeze().astype(np.uint16)
    if data.dtype.byteorder == '>':  # big endian
        data = data.astype(data.dtype.newbyteorder('<'))
    return data

def _create_base_dataset(dcm_name: str, dcm_sex: str, dcm_laterality: str, 
                        dcm_details: str, mm: list[float], axis: int, 
                        thickness_mm: float) -> pydicom.dataset.Dataset:
    """Create base DICOM dataset with common tags."""
    ds = pydicom.dataset.Dataset()
    
    # Patient information
    ds.PatientID = dcm_name
    ds.PatientName = dcm_name
    ds.PatientSex = dcm_sex
    ds.ImageLaterality = dcm_laterality
    ds.AccessionNumber = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesDescription = dcm_details

    # Study information
    ds.PatientBirthDate = "19800101"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.StudyID = dcm_details
    ds.StudyDescription = dcm_details

    # Image properties
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.Modality = "MG"
    ds.ViewPosition = "CC"
    ds.PixelSpacing = [mm[i] for i in range(3) if i != axis][:2]
    ds.SliceThickness = thickness_mm
    ds.SpacingBetweenSlices = mm[axis]
    ds.WindowWidth = "1150"
    ds.WindowCenter = "625"
    ds.ConversionType = "WSD"
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesNumber = None

    # Pixel data properties
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15        
    ds.PixelRepresentation = 0
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Rescale properties
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.RescaleType = "US"

    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.FileMetaInformationGroupLength = 238
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.ImplementationVersionName = f'pydicom {pydicom.__version__}'
    file_meta.SourceApplicationEntityTitle = 'pydicom'
    ds.file_meta = file_meta
    ds.preamble = b'\x00' * 128

    return ds

def _write_slice(slice_data: np.ndarray, slice_idx: int, path: Path, 
                file_prefix: str, ndigits: int, base_ds: pydicom.dataset.Dataset) -> None:
    """Write a single slice to DICOM file."""
    
    dsc = _create_slice_dataset(base_ds, slice_data, slice_idx)
    
    output_filename = path / f"{file_prefix}{slice_idx+1:0{ndigits}d}.dcm"
    pydicom.filewriter.dcmwrite(output_filename, dsc)

def _create_slice_dataset(base_ds: pydicom.dataset.Dataset, 
                         slice_data: np.ndarray, slice_idx: int) -> pydicom.dataset.Dataset:
    """Create dataset for a single slice."""
    dsc = base_ds.copy()
    dsc.SOPInstanceUID = pydicom.uid.generate_uid()
    
    # Set study time
    now = datetime.now()
    dsc.StudyTime = str(float(now.strftime('%H%M%S')) + (now.microsecond / 1_000_000))
    
    # Set pixel data
    dsc.Rows = slice_data.shape[0]
    dsc.Columns = slice_data.shape[1]
    dsc.PixelData = slice_data.tobytes()

    # Spatial information
    dsc.InstanceNumber = slice_idx + 1
    dsc.ImagePositionPatient = [0.0, 0.0, slice_idx * dsc.SpacingBetweenSlices]
    dsc.SliceLocation = slice_idx * dsc.SpacingBetweenSlices
    
    return dsc


class ScaleTo12Bit:
    """
    Scale data to 12-bit range for DICOM output.
    
    Assumes input data in beta scale.
    Standard normalization for 16-bit TIFF output in BCT: (min-in max-in max-out) = (5.0e-11 7.0e-10 4095)
    Image data type will be uint16.
    Scalar range will be 12 bit.
    Default values are for 32 keV BCT.
    """
    
    def __init__(self, in_min: float = 5.0e-11, in_max: float = 7.0e-10, out_max: int = 4095):
        """
        Initialize scaler with desired input/output ranges.
        
        Args:
            in_min (float): Minimum input value
            in_max (float): Maximum input value  
            out_max (int): Maximum output value (default 4095 for 12-bit)
        """
        self.in_min = in_min
        self.in_max = in_max
        self.out_max = out_max

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Scale input data to 12-bit range.
        
        Args:
            data (np.ndarray): Input data in beta scale
            
        Returns:
            np.ndarray: Scaled data as uint16
        """
        logger.debug(f"Scaling data to 12-bit range")
        logger.debug(f"Input data range: [{data.min():.2e}, {data.max():.2e}]")
        
        # Convert to float64 for precision
        data = data.astype(np.float64)
        
        # Scale to 12-bit range
        data = (data - self.in_min) / (self.in_max - self.in_min) * self.out_max
        
        # Clamp to valid range and convert to uint16
        data = np.clip(data, 0, self.out_max).astype(np.uint16)
        
        logger.debug(f"Output data range: [{data.min()}, {data.max()}]")
        
        return data