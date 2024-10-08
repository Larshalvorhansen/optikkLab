import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def RMSE(real_data, predicted_data):
    assert len(real_data) == len(predicted_data)
    n = len(real_data)
    sum_values = []
    for i in range(len(real_data)):
        values = np.power(real_data[i] - predicted_data[i],2)
        sum_values.append(values)
    
    rmse = np.sqrt((1/n)*np.sum(sum_values))
    return rmse 

def coefficient_calc(polyorder,observed_pixel,known_refrence_lambda):
    x_cal    = 0
    y_cal    = 0
    xy_cal   = 0
    x_2_y_cal =0
    x_2_cal  = 0
    x_3_cal  = 0
    x_4_cal  = 0
    x_4_cal  = 0
    for i in range(len(observed_pixel)):
        x_cal += observed_pixel[i]
        y_cal += known_refrence_lambda[i]
        xy_cal += x_cal * y_cal
        x_2_cal += x_cal * x_cal
        if (polyorder == 2):
            x_3_cal += x_cal * x_cal * x_cal
            x_4_cal += x_3_cal *x_cal
            x_2_y_cal += x_2_cal *y_cal
    if(polyorder == 2):
        return x_cal,y_cal,xy_cal,x_2_cal,x_3_cal,x_4_cal,x_2_y_cal
    else:
        x_cal,y_cal,xy_cal,x_2_cal



def plynominal_model(polyorder,observed_pixel, known_refrence_lambda):
    n = len(observed_pixel)
    
    #sums to calculate coefficiens
    sums = coefficient_calc(polyorder,observed_pixel,known_refrence_lambda)

    if (polyorder == 1):
        x_cal,y_cal,xy_cal, x_2_cal = sums
        a_1 = (n * xy_cal - x_cal * y_cal)/(n * x_2_cal - np.power(x_cal,2))
        a_0 = (y_cal - a_1 * x_cal) / n
        y_pred = a_1 *np.array(observed_pixel) +a_0
        return y_pred

    else:
        x_cal, y_cal, xy_cal, x_2_cal, x_3_cal, x_4_cal, x_2_y_cal = sums
        A = np.array([[len(observed_pixel), x_cal, x_2_cal],
                       [x_cal, x_2_cal, x_3_cal],
                       [x_2_cal, x_3_cal, x_4_cal]])
        
        B = np.array([y_cal, xy_cal, x_2_y_cal])
        
        a_0, a_1, a_2 = np.linalg.solve(A, B)
        
        y_pred = a_2 * np.array(observed_pixel)**2 + a_1 * np.array(observed_pixel) + a_0
        return y_pred

def log_model(scale_factor,observed_pixel,known_refrence_lambda):
    return scale_factor * np.exp(observed_pixel* known_refrence_lambda)

def log_fit(observed_pixel,known_refrence_lambda):
    x = np.array(observed_pixel)
    y = np.array(known_refrence_lambda)

    factors, curve = curve_fit(exponential_model, x,y)
    a,b = factors

    y_pred = log_fit(x,a,b)
    return y_pred



########## CALL ON FUCTIONS #######
# polynimal_predictions = plynominal_model(1,pixel_position, known_refrence) 
# logarithminc_predictions = log_fit(pixel_position,known_refrence)
# error = RMSE(polynimal_predictionspredictions, known_refrence)


