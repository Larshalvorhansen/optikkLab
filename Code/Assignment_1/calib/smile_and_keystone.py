''' Code for detecting and estimating smile and keystone, and calculating
the correction coefficients, as well as applying the correction.

Built on third go on detection of smile and keystone.
Uses one ar image and one hg image, masks them (0 and 1) and combines them.
Centre of masked points are found and used as GCPs and sorted onto grid.
GCPs are used to estimate smile and keystone in the system.

Updated: moving adjustable parameters to be input in the functions.
        
2021-11-24      MBH     - Created file.
2024-06-20      SB      - Modified file for TTK4265.

Note: 
This file is based on the calibration functions from the HYPSO-1 project.
Not all functions will be necessary for the TTK4265 project, but they are
included for completeness. 

You may need to edit these files to fit your project.
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from calibration_functions import pixel_to_wavelength

def estimate_smile_and_keystone( im, spectral_coeffs, y_start=0, image_height=-1, x_start=0, x_stop=-1, plot=False ):
    ''' Uses subfunctions to find GCPs and sort them on grid, then estimates
    the smile and keystone from this. Prints values and shows plots.
    Returns the fit coefficients for smile and keystone.'''
    
    # Set sizes 
    if image_height == -1:
        image_height, image_width = im.shape 
    else:
        image_width = x_stop - x_start
    w = np.linspace(x_start,image_width+x_start,image_width)
    middle_w = int(image_width/2) + x_start
    h = np.linspace(y_start,image_height+y_start,image_height)
    middle_h = int(image_height/2) + y_start

    # Get GCP 
    x_coord, y_coord = get_centre_coordinates_from_masked_image(im)
    gcp_nparr = sort_centre_coordinates_to_grid(x_coord, y_coord, im)

    num_cols = gcp_nparr.shape[1]
    num_rows = gcp_nparr.shape[0]
    
    ##-- Estimate smile and keystone lines --##
    # Make horisontal line fit, keystone, first order
    hor_fit_coeffs = []
    for i in range(num_rows):  
        row = gcp_nparr[i] 
        row_coords = []    
        for j in range(num_cols):
            coord = row[j]
            # Only use actual coordinates, not (-1, -1)
            if coord[0] > 0:
                row_coords.append(coord)
        row_coords_nparr = np.array(row_coords)
        hor_coeff = np.polyfit(row_coords_nparr[:,0], row_coords_nparr[:,1], 1)
        hor_fit_coeffs.append(hor_coeff)

    # Make vertical line fit, smile, second order
    ver_fit_coeffs = []
    for i in range(num_cols):  
        col = gcp_nparr[:,i]
        col_coords = []
        for j in range(num_rows):
            coord = col[j]
            # Only use actual coordinates, not (-1, -1)
            if coord[0] > 0:
                col_coords.append(coord)
        col_coords_nparr = np.array(col_coords)
        ver_coeff = np.polyfit(col_coords_nparr[:,1], col_coords_nparr[:,0], 2)
        ver_fit_coeffs.append(ver_coeff)
        
    hor_fit_coeffs = np.array(hor_fit_coeffs)
    ver_fit_coeffs = np.array(ver_fit_coeffs)


    ##-- Quanitfy smile --##

    # Should only use selected smile range
    if image_height != -1:
        num_lines_selected = 0
        min_w = pixel_to_wavelength(x_start, spectral_coeffs)
        max_w = pixel_to_wavelength(x_stop, spectral_coeffs)
        for smile_coeff in ver_fit_coeffs: 
                ver_fit_func = np.poly1d(smile_coeff)
                mid_value = ver_fit_func(middle_h)                           
                # Calculate wavelength 
                mid_w = pixel_to_wavelength(mid_value, spectral_coeffs)
                if mid_w > min_w and mid_w < max_w:
                    num_lines_selected += 1

    all_shift_mid_max = []
    all_shift_min_max = []
    for i in range(num_lines_selected):
        smile_coeff = ver_fit_coeffs[i]
        smile_func = np.poly1d(smile_coeff)                
        smile_line = smile_func(h)
        # Find values
        max_val = max(smile_line)
        mid_val = smile_func(middle_h)
        min_val = min(smile_line)
        diff_mid_max = max_val - mid_val
        diff_min_max = max_val - min_val
        # Add values
        all_shift_mid_max.append(diff_mid_max)
        all_shift_min_max.append(diff_min_max)   
    
    # avg_shift_mid_max = np.average(all_shift_mid_max) # Avg shift 
    avg_shift_min_max = np.average(all_shift_min_max) # Avg shift 
    # max_shift_mid_max = np.max(all_shift_mid_max) # Max shift 
    max_shift_min_max = np.max(all_shift_min_max) # Max shift 
    
    print("Smile:")
    print("Avg shift min/max: %.2f" %avg_shift_min_max)
    print("Max shift min/max: %.2f" %max_shift_min_max)
    
    ##-- Quantify keystone --##
    all_shift_min_max = []
    for keystone_coeff in hor_fit_coeffs:
        keystone_func = np.poly1d(keystone_coeff)                
        keystone_line = keystone_func(w)
        # Find values
        max_val = max(keystone_line)
        min_val = min(keystone_line)
        diff_min_max = max_val - min_val
        # Add values
        all_shift_min_max.append(diff_min_max)  
    
    avg_shift_min_max = np.average(all_shift_min_max) # Avg shift fra min til max
    max_shift_min_max = np.max(all_shift_min_max) # Max shift fra min til max
    
    print("Keystone:")
    print("Avg shift min/max: %.2f" %avg_shift_min_max)
    print("Max shift min/max: %.2f" %max_shift_min_max)
    
    ##-- Smile and keystone plots --##
    if plot:
        # Plot smile
        fig, ax = plt.subplots()
        cm = plt.get_cmap('viridis')
        NUM_COLORS = 20
        ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS, 0, -1)])
        i = 0
        if image_height == -1:
            for i in range(num_lines_selected):
                smile_coeff = ver_fit_coeffs[i]
                ver_fit_func = np.poly1d(smile_coeff)
                mid_value = ver_fit_func(middle_h)                           
                # Calculate wavelength for use in legend
                mid_w = pixel_to_wavelength(mid_value, spectral_coeffs)
                # Plot info from this line/ these coeffs
                line = ver_fit_func(h) - mid_value
                plt.plot(h, line, label='%0.2f nm' %mid_w)
                i += 1
        else:
            for smile_coeff in ver_fit_coeffs: 
                ver_fit_func = np.poly1d(smile_coeff)
                mid_value = ver_fit_func(middle_h)                           
                # Calculate wavelength for use in legend
                mid_w = pixel_to_wavelength(mid_value, spectral_coeffs)           
                # Plot info from this line/ these coeffs
                line = ver_fit_func(h) - mid_value
                plt.plot(h, line, label='%0.2f nm' %mid_w)
                i += 1

        # Plot middle line for reference  
        plt.axvline(x=middle_h, color='silver',linestyle='--', linewidth=1)
        plt.title('Spectral shift due to smile, 400 to 800 nm')
        plt.xlabel('Spatial axis [pixel]')
        plt.ylabel('Spectral pixel shift [pixel]')
        plt.legend(bbox_to_anchor=(1, 0), loc='lower left', ncol=1) 
        fig.tight_layout()
        # plt.savefig('geometric/smile_400-to-800-nm.png', bbox_inches = 'tight')
    
        # Plot keystone
        fig, ax = plt.subplots()
        cm = plt.get_cmap('viridis')
        NUM_COLORS = 20
        ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS, 0, -1)])
        i = 0
        for key_coeff in hor_fit_coeffs: 
            key_func = np.poly1d(key_coeff)
            mid_value = key_func(middle_w)                               
            # Plot info from this line/ these coeffs
            line = key_func(w) - mid_value
            plt.plot(w, line, label='Start pixel: %i' 
                    %key_coeff[1])
            i += 1
        plt.title('Pixel shift due to keystone, 400 to 800 nm')
        plt.xlabel('Spectral pixel [pixel]')
        plt.ylabel('Spatial axis pixel shift [pixel]')
        plt.legend(bbox_to_anchor=(1, 0), loc='lower left', ncol=1) 
        fig.tight_layout()
        # plt.savefig('geometric/keystone_400-to-800-nm.png', bbox_inches = 'tight')
    
    return gcp_nparr, hor_fit_coeffs, ver_fit_coeffs


def model_smile_and_keystone( im, spectral_coeffs, model_filenames, plot=False ):
    ''' Detects and estimates smile and keystone in a single frame by finding
    GCPs which are used to create a reference grid. This is further used to 
    model the full distortion, and model coefficients are saved to file.
    '''
    image_height, image_width = im.shape  
    
    ## - Detect peaks and find GCPs - ##
    info = estimate_smile_and_keystone(im, spectral_coeffs, plot)
    gcp_nparr, hor_fit_coeffs, ver_fit_coeffs = info
    
    ## - Make reference frame - ##
    # Find smile line nr closest to mid wavelength using first horizontal line
    mid_w_pixel = int(image_width/2)
    mid_smile_index = 0
    for i in range(1, gcp_nparr.shape[1]):
        w_pixel = gcp_nparr[0, i, 0]
        if w_pixel >= mid_w_pixel: # Assuming increasing index
            diff = abs(mid_w_pixel - w_pixel)
            last_diff = abs(mid_w_pixel - gcp_nparr[0, i-1, 0])
            if diff < last_diff: # Check if this or last one is closest
                mid_smile_index = i
            else:
                mid_smile_index = i-1
            break
    mid_smile_line = gcp_nparr[:, mid_smile_index]

    # Find middle keystone line in mid smile line
    mid_h_pixel = int(image_height/2)
    for i in range(len(mid_smile_line)):
        h_pixel = mid_smile_line[i, 1] 
        if h_pixel >= mid_h_pixel: # Assuming increasing index
            diff = abs(mid_h_pixel - h_pixel)
            last_diff = abs(mid_h_pixel - mid_smile_line[i-1, 1])
            if diff < last_diff: # Check if this or last one is closest    
                mid_keystone_index = i
            else:
                mid_keystone_index = i-1
            break
    mid_keystone_line = gcp_nparr[mid_keystone_index, :]
            
    # Reference points
    ref_nparr = np.empty_like(gcp_nparr)*0
    for i in range(ref_nparr.shape[0]):
        ref_nparr[i,:,0] = mid_keystone_line[:,0]
    for i in range(ref_nparr.shape[1]):
        ref_nparr[:,i,1] = mid_smile_line[:,1]


    ## - Find model coefficients - ##

    # Using estimated smile and keystone lines to estimate point locations
    # Basically smooths out the position of the gcp points
    gcp_nparr_est = np.empty_like(gcp_nparr)
    for w in range(gcp_nparr_est.shape[1]):
        for h in range(gcp_nparr_est.shape[0]):
            y = mid_smile_line[h][1]
            x_est = (y*y*ver_fit_coeffs[w,0] + y*ver_fit_coeffs[w,1] + ver_fit_coeffs[w,2])
            gcp_nparr_est[h,w,0] = x_est
    for h in range(gcp_nparr_est.shape[0]):
        for w in range(gcp_nparr_est.shape[1]):
            x = mid_keystone_line[w][0]
            y_est = x*hor_fit_coeffs[h,0] + hor_fit_coeffs[h,1]
            gcp_nparr_est[h,w,1] = y_est

    # Convert to list with float32 coords (required input)
    gcp_list = np.float32(gcp_nparr_est.reshape(-1, gcp_nparr_est.shape[-1]))
    ref_list = np.float32(ref_nparr.reshape(-1, ref_nparr.shape[-1]))

    # Find coefficients
    A_hat, B_hat = find_model_coeffs(gcp_list, ref_list) 
    
    # Write coefficients to file
    [A_hat_filename, B_hat_filename] = model_filenames
    np.savetxt(A_hat_filename, A_hat, delimiter=",")
    np.savetxt(B_hat_filename, B_hat, delimiter=",") 
        
    return None


def correct_smile_and_keystone( im, filenames ):
    ''' Corrects image simply by running the function transform() with the 
    image and the model coefficients. '''

    [A_hat_filename, B_hat_filename] = filenames
    A_hat = A_hat_filename # TODO: Put correct fileoutput here
    B_hat = B_hat_filename # TODO: Put correct fileoutput here

    # Correct image
    im_corr = transform(A_hat, B_hat, im)

    return im_corr


#########################
### SMALLER FUNCTIONS ###
#########################

def get_centre_coordinates_from_masked_image( im ):
    ''' Input frame with masked points/blobs with values 0 and 1. OpenCV works
    with value range 0-255, so input values are scaled. Contours of points/
    blobs are detected, and the centre position calculated.
    Returns two lists with x- and y-coordinates, respectively. 
    
    From example: https://learnopencv.com/find-center-of-blob-centroid-using-
    opencv-cpp-python/'''

    image_8bit = np.uint8(im * 255)
    threshold_level = 127
    _, binarized = cv2.threshold(image_8bit, threshold_level, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    min_blob_size = 10 # [ADJUST THIS]

    x_coord_list = []
    y_coord_list = []
    for c in contours:
        # Calculate moments for each contour
        M = cv2.moments(c)

        # Calculate x,y coordinate of center
        if M["m00"] > min_blob_size:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            x_coord_list.append(int(cX))
            y_coord_list.append(int(cY))

    return x_coord_list, y_coord_list


def sort_centre_coordinates_to_grid( x_coord, y_coord, im ):
    ''' Sorts list of points onto grid. First sorts them into rows, using 
    parameter height_distance. Then removes additional points that are too
    close to each other, using parameter blob_radius. Sorts into numpy array, 
    sets missing coords to (-1, -1).

    Returns numpy array with points on grid.'''

    blob_radius = 10 # Limit, only one point inside blob [ADJUST THIS] 
    height_distance = 20 # Limit, more than this = new row [ADJUST THIS]
    row_list = []
    coord_matrix = []
    num_smile_lines_list = []
    
    # Initialize with first coords
    x_last = x_coord[0]
    y_last = y_coord[0]
    coord = [x_last, y_last]
    row_list.append(coord)

    # Sort coordinates into matrix, row by row
    for i in range(1, len(x_coord)):
        x = x_coord[i]
        y = y_coord[i]
        coord = [x, y]

        # Check height, if same (within blob) add to same row
        # and if last element go directly to else statement
        if abs(y - y_last) < height_distance and i != len(x_coord)-1: 
            row_list.append(coord)
            y_last = y

        # New row detected
        else: 
            # Add last element if this is what triggered else statement
            if i == len(x_coord)-1:
                row_list.append(coord)

            # Sort points in row on x-coords (rising order)
            row_list = sorted(row_list, key=lambda x: x[0])

            # Remove any points that are too close to each other
            x_last = row_list[0][0]
            row_list_points_removed = []
            row_list_points_removed.append(row_list[0])
            for j in range(1, len(row_list)):
                x = row_list[j][0]
                diff = abs(x - x_last)
                if diff > blob_radius:
                    row_list_points_removed.append(row_list[j])
                else:
                    print('Removing a GCP point, was inside same blob.')
                x_last = x

            # Count how many points were detected
            num_smile_lines = len(row_list_points_removed)
            num_smile_lines_list.append(num_smile_lines)

            # Add row to matrix and initialize
            coord_matrix.append(row_list_points_removed)
            row_list = []

            # This point belongs to next row
            row_list.append(coord) 
            y_last = y

    # Sort coords into numpy array, set missing points to -1
    num_cols = max(num_smile_lines_list) # Number of smile lines
    full_row_index = num_smile_lines_list.index(num_cols)
    ref_row = coord_matrix[full_row_index]
    num_rows = len(coord_matrix) # Number of keystone lines
    coord_nparr = np.full([num_rows, num_cols, 2], -1)

    # Compare all elements with ref row
    for i in range(num_rows):
        row = coord_matrix[i]
        j = 0
        lim = 15 # Limit, maximum smile offset for coords to be in same line
        for ref_j in range(num_cols):
            elem = row[j][0]
            ref_elem = ref_row[ref_j][0]

            diff = abs(elem - ref_elem)
            if diff < lim:
                coord_nparr[i, ref_j] = row[j]
                j += 1 

    # Remove rows and columns with too few points
    min_keystone_lines = 5 # TODO: ADJUST THIS
    min_smile_lines = 5 # TODO: ADJUST THIS

    # Remove rows 
    del_count = 0
    for i in range(num_rows):
        row = coord_nparr[i]
        elem_count = 0
        for j in range(num_cols):
            coord = row[j]
            if coord[0] > 0:
                elem_count += 1
        if elem_count < min_smile_lines:
            coord_nparr = np.delete(coord_nparr, i, 0) # Delete this row
            del_count += 1
            print('Too few good points found in row, row removed.')
    num_rows = num_rows - del_count

    # Remove columns
    del_count = 0
    for i in range(num_cols):
        i = i - del_count
        col = coord_nparr[:,i]
        elem_count = 0
        for j in range(num_rows):
            coord = col[j]
            if coord[0] > 0:
                elem_count += 1
        if elem_count < min_keystone_lines:
            coord_nparr = np.delete(coord_nparr, i, 1) # Delete this column
            del_count += 1
            print('Too few good points found in column, column removed.')
    num_cols = num_cols - del_count

    return coord_nparr


def find_model_coeffs( gcp, ref ):
    ''' Uses the detected intersection points (here known as ground control
    points (GCPs)) in the frame and the generated reference points to make a 
    model of the distortion in the frame, as suggested in Lawrence (2003).
    
    Polynomial distortion model
    x = a_00 + a_10*x_ref + a_01*y_ref + a_11*x_ref*y_ref + a_20*x_ref*x_ref 
        + a_02*y_ref*y_ref
    y = b_00 + b_10*x_ref + b_01*y_ref + b_11*x_ref*y_ref + b_20*x_ref*x_ref 
        + b_02*y_ref*y_ref
    
    Matrix form:
    X = WA
    Y = WB
    where W is a matrix consisting of vectors w for each point,
    w = [1 x_ref1 y_ref1 x_ref1*y_ref1 x_ref1*x_ref1 y_ref1*y_ref1]
    
    Pseudo-inverse solutions
    A_hat = (np.transpose(W)*W)^(-1)*np.transpose(W)*X
    B_hat = (np.transpose(W)*W)^(-1)*np.transpose(W)*Y
    '''
    
    # Make W for calculating A and B (based on GCP and ref points)
    W = []
    for x_ref, y_ref in ref:
        w = [1, x_ref, y_ref, x_ref*y_ref, x_ref*x_ref, y_ref*y_ref]
        W.append(w)
    W_nparr = np.array(W)
    W_nparr_transposed = np.transpose(W_nparr)   
   
    # Split into x- and y-coordinates
    X, Y = map(list, zip(*gcp))
    
    wtw = np.dot(W_nparr_transposed, W_nparr)
    wtw_inv = np.linalg.pinv(wtw) # Using pseudo-inverse to avoid singularity issues

    # Calculate A_hat
    # A_hat = (W^T * W)^(-1) * W^T * X
    wt_x = np.dot(W_nparr_transposed, X)
    A_hat = np.dot(wtw_inv, wt_x)
    
    # Calculate B_hat 
    # B_hat = (W^T * W)^(-1) * W^T * Y
    wt_y = np.dot(W_nparr_transposed, Y)
    B_hat = np.dot(wtw_inv, wt_y)
    
    return A_hat, B_hat


def transform( A_hat, B_hat, im ):
    ''' Uses model coefficients from the detected intersection points, and 
    then corrects by resampling, as suggested in Lawrence (2003).
    
    As of now, the function scipy.ndimage.map_coordinates is used for 
    resampling, this should probably be changed if only parts of the frame
    should be corrected at a time.
    '''    
    
    # Set sizes
    image_height, image_width = im.shape
    
    # Make all ref coords
    ref_all = []
    for x_ref in range(image_width):
        for y_ref in range(image_height):
            ref_all.append([x_ref, y_ref])
    
    # Make W for all ref coords
    W_all = []
    for x_ref, y_ref in ref_all:
        w = [1, x_ref, y_ref, x_ref*y_ref, x_ref*x_ref, y_ref*y_ref]
        W_all.append(w)
    W_all_nparr = np.array(W_all)
    
    # Apply equations (no noise)
    new_x_ref_vec = np.matmul(W_all_nparr, A_hat) 
    new_y_ref_vec = np.matmul(W_all_nparr, B_hat)

    # Use map_coordinates to do interpolation and filling in values
    corr_im_list = ndimage.map_coordinates(im, [new_y_ref_vec, new_x_ref_vec])
    corr_im = np.reshape(corr_im_list, [image_width, image_height])
    corr_im = np.transpose(corr_im)
    
    return corr_im

