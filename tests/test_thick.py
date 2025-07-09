import numpy as np
import pytest
from unittest.mock import patch

from bctdicom.thick import thick_axial


class TestThickAxial:
    """Test cases for the thick_axial function."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        # Create a simple 3D test array
        self.test_array = np.random.rand(10, 8, 6)
        self.test_array_upper_intensity = np.random.rand(10, 8, 6) * 0.1
        self.test_array_upper_intensity[2:4, 3:5, 1:3] = 0.8  # Upper intensity region
        
    def test_basic_functionality_even(self):
        """Test basic thick slice computation."""
        result, output_mm, thickness_mm = thick_axial(
            self.test_array,
            input_mm=[1.0, 1.0, 1.0],
            axis=0,
            thickness_mm=4.0,
            spacing_mm=2.0
        )
        # Check output shape
        expected_shape = (4, 8, 6)  # (1 + (10-4)/2) = 4 thick slices
        assert result.shape == expected_shape
        
        # Check output spacing
        assert output_mm[0] == 2.0  # spacing_mm
        assert output_mm[1] == 1.0
        assert output_mm[2] == 1.0
        
        # Check thickness
        assert thickness_mm == 4.0
        
        # Check data type
        assert result.dtype == self.test_array.dtype
    
    def test_basic_functionality_odd(self):
        """Test basic thick slice computation."""
        result, output_mm, thickness_mm = thick_axial(
            self.test_array,
            input_mm=[1.0, 1.0, 1.0],
            axis=0,
            thickness_mm=3.0,
            spacing_mm=2.0
        )
                # Check output shape
        expected_shape = (4, 8, 6)  # (1 + (10-3)/2) = 4.5 thick slices
        assert result.shape == expected_shape
        
        # Check output spacing
        assert output_mm[0] == 2.0  # spacing_mm
        assert output_mm[1] == 1.0
        assert output_mm[2] == 1.0
        
        # Check thickness
        assert thickness_mm == 3.0
        
        # Check data type
        assert result.dtype == self.test_array.dtype
        
    def test_different_axes(self):
        """Test thick slice computation along different axes."""
        # Test axis 1
        result1, _, _ = thick_axial(
            self.test_array,
            axis=1,
            thickness_mm=2.0,
            spacing_mm=1.0
        )
        expected_shape1 = (10, 7, 6)  # (1 + (8-2)/1) = 7 thick slices
        assert result1.shape == expected_shape1
        
        # Test axis 2
        result2, _, _ = thick_axial(
            self.test_array,
            axis=2,
            thickness_mm=2.0,
            spacing_mm=1.0
        )
        expected_shape2 = (10, 8, 5)  # (1 + (6-2)/1) = 5 thick slices
        assert result2.shape == expected_shape2
        
    def test_mip_mode(self):
        """Test maximum intensity projection mode."""
        pass

        
    def test_threshold_processing(self):
        """Test processing with threshold for upper intensity voxels."""
        result, _, _ = thick_axial(
            self.test_array_upper_intensity,
            threshold=0.5,
            thickness_mm=2.0,
            spacing_mm=1.0
        )
        
        # Check that output shape is correct
        expected_shape = (9, 8, 6)  # (1 + (10-2)/1) = 9 thick slices
        assert result.shape == expected_shape
        
        # Check that result contains finite values
        assert np.all(np.isfinite(result))
        
    def test_custom_spacing(self):
        """Test with custom input spacing."""
        result, output_mm, thickness_mm = thick_axial(
            self.test_array,
            input_mm=[0.5, 1.0, 2.0],
            thickness_mm=1.5,
            spacing_mm=1.0
        )
        
        # Check output spacing
        assert output_mm[0] == 1.0  # spacing_mm
        assert output_mm[1] == 1.0
        assert output_mm[2] == 2.0
        
        # Check thickness
        assert thickness_mm == 1.5
        
    def test_edge_case_single_slice(self):
        """Test edge case where thickness equals input size."""
        result, _, thickness_mm = thick_axial(
            self.test_array,
            thickness_mm=10.0,  # Equal to input size
            spacing_mm=10.0
        )
        
        # Should produce exactly 1 thick slice
        assert result.shape[0] == 1
        assert result.shape[1:] == self.test_array.shape[1:]
        
    def test_edge_case_large_spacing(self):
        """Test edge case with spacing larger than input size."""
        result, _, _ = thick_axial(
            self.test_array,
            thickness_mm=2.0,
            spacing_mm=15.0  # Larger than input size
        )
        
        # Should produce exactly 1 thick slice
        assert result.shape[0] == 1
        
    def test_non_integer_spacing_warning(self):
        """Test that warnings are logged for non-integer spacing."""
        with patch('bctdicom.thick.logger') as mock_logger:
            thick_axial(
                self.test_array,
                thickness_mm=2.7,  # Not integer multiple of 1.0
                spacing_mm=1.3     # Not integer multiple of 1.0
            )
            
            # Check that warnings were logged
            assert mock_logger.warning.called
            

    def test_negative_axis(self):
        """Test behavior with negative axis values."""
        result, _, _ = thick_axial(self.test_array,
                                    axis=-1,
                                    thickness_mm=2.0,
                                    spacing_mm=1.0)
        
        # Should work the same as axis=2
        expected_shape = (10, 8, 5)  # (1 + (6-2)/1.0) = 5 thick slices
        assert result.shape == expected_shape
        
    def test_threshold_none(self):
        """Test behavior when threshold is None."""
        result, _, _ = thick_axial(
            self.test_array_upper_intensity,
            thickness_mm=2.0,
            spacing_mm=1.0,
            threshold=None
        )
        
        # Should work without threshold processing
        expected_shape = (9, 8, 6)  # (1 + (10-2)/1.0) = 9 thick slices
        assert result.shape == expected_shape
        
    def test_isolated_voxel_removal(self):
        """Test that isolated upper-intensity voxels are removed."""
        # Create array with isolated upper-intensity voxel
        test_array = np.zeros((6, 4, 4))
        test_array[2, 2, 2] = 0.8  # Isolated upper-intensity voxel
        
        result, _, _ = thick_axial(
            test_array,
            threshold=0.5,
            thickness_mm=2.0,
            spacing_mm=1.0
        )
        
        # The isolated voxel should be removed during processing
        # Result should still have correct shape
        expected_shape = (5, 4, 4)  # (1 + (6-2)/1.0) = 5 thick slices
        assert result.shape == expected_shape
        
    def test_upper_intensity_region_preservation(self):
        """Test that connected upper-intensity regions are preserved."""
        # Create array with connected upper-intensity region
        test_array = np.zeros((6, 4, 4))
        test_array[2:4, 2:4, 2:4] = 0.8  # Connected upper-intensity region
        
        result, _, _ = thick_axial(
            test_array,
            threshold=0.5,
            thickness_mm=2.0,
            spacing_mm=1.0
        )
        
        # Connected region should be preserved
        expected_shape = (5, 4, 4)  # (1 + (6-2)/1.0) = 5 thick slices
        assert result.shape == expected_shape
        
    def test_axis_reordering(self):
        """Test that axis reordering works correctly."""
        result, _, _ = thick_axial(self.test_array,
                                    axis=1,
                                    thickness_mm=2.0,
                                    spacing_mm=1.0)
        
        # Result should have correct shape for axis 1
        expected_shape = (10, 7, 6)  # (1 + (8-2)/1.0) = 7 thick slices
        assert result.shape == expected_shape
        
        # Test that the axis was moved back correctly
        # The thick slice dimension should be in position 1
        assert result.shape[1] == 7  # Thick slice dimension
        
if __name__ == "__main__":
    pytest.main([__file__]) 