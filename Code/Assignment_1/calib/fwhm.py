''' To calculate FWHM.

2021-11-23      MBH     - Created file.
2024-06-20      SB      - Modified file for TTK4265.

Note: 
This file is based on the calibration functions from the HYPSO-1 project.
Not all functions will be necessary for the TTK4265 project, but they are
included for completeness. 

You may need to edit these files to fit your project.
'''

import imageio as iio
import scipy as sp
from typing import List, Tuple

from calib.calibration_functions import pixel_to_wavelength

def calc_fwhm(files: List[str], lines: List[int], coeff: float) -> Tuple[List[List[float]], List[int]]:
    """
    Calculate the full width at half maximum (FWHM) for each peak in the given lines of the images.
    
    Parameters:
    - files (list): A list of file paths to the images.
    - lines (list): A list of line indices to calculate the FWHM.
    - coeff (float): The coefficient for converting pixel position to wavelength.
    
    Returns:
    - all_fwhm_list (list): A list of lists, where each inner list contains the FWHM values for each peak in a line.
    - skip_list (list): A list of line indices that were skipped due to an incorrect number of peaks.
    """
    
    # Open data
    all_fwhm_list: List[List[float]] = []
    skip_list: List[int] = []
    for line in lines:
        peaks_width_nm: List[float] = []
        for file in files:
            im = iio.imread(file)[:,::-1]
        
            # Detect position of peaks # TODO: Change if needed
            max_val = max(im[line])
            peak_height = 0.1*max_val # Minimum value of peak
            distance = 18 # Minimum distance between peaks
            smooth_line = im[line] # Consider smoothing the line.
            peaks_pos, peaks_height_dict = sp.signal.find_peaks(smooth_line, peak_height, None, distance)    
            peaks_height = list(peaks_height_dict.values())[0]
            
            # Find width at half maximum           
            results_half = sp.signal.peak_widths(smooth_line, peaks_pos, rel_height=0.5) # At half maximum
            peaks_width = results_half[0]
            
            # Convert width from pixel to nm
            num_peaks = len(peaks_height)
            for peak in range(num_peaks):
                
                # Find pixel pos of half width on each side
                half_width = peaks_width[peak]/2
                min_w_pos = peaks_pos[peak] - half_width
                max_w_pos = peaks_pos[peak] + half_width
                
                # Convert half width pos to nm 
                min_w = pixel_to_wavelength(min_w_pos, coeff)
                max_w = pixel_to_wavelength(max_w_pos, coeff)
                
                # Width in nm
                width_nm = max_w - min_w
                peaks_width_nm.append(float(width_nm))
            
        if len(peaks_width_nm) == 15: # TODO: change if needed
            all_fwhm_list.append(peaks_width_nm)
        else:
            skip_list.append(line)
    
    return all_fwhm_list, skip_list