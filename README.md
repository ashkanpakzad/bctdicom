# BCT DICOM Export Tool
A Python application for processing and exporting Breast CT (BCT) images to DICOM format for clinical use.

Defaults are set for 32 keV X-ray Xineos detected images with 0.1 mm^3 voxel spacing in beta scale.

## Features

### Image Processing
- Create mean thick slices from thin BCT slices along any axis
- Configurable slice thickness and spacing
- Maximum Intensity Projection (MIP) instead of mean option
- Automatic threshold-based calcification detection and enhancement
- Flexible axis flipping and swapping for proper orientation

### DICOM Export
- Export BCT images to standard DICOM format
- 12-bit depth scaling optimized for BCT contrast
- Configurable DICOM metadata including:
  - Patient name/ID
  - Patient sex
  - Image laterality (left/right)
  - Study/series details
  - Slice thickness
  - Voxel spacing

### File Format Support
- Input formats:
  - NRRD (.nrrd)
  - HDF5 (.h5, .hdf)
- Output formats:
  - DICOM (.dcm)
  - NRRD (.nrrd) 
  - HDF5 (.h5, .hdf)

### Usage
- Simple command-line interface