#--------------- Importing key libraries and modules ---------------#
import time
from datetime import datetime, timedelta

import numpy as np
from uncertainties import ufloat # for managing errors in the data
from uncertainties import unumpy
import scipy.constants as con

import data_functions as data_func
import calc_functions as calc_func

def within_error(m1, m2):
    """
    Function to return whether two measurements, m1 and m2, agree within error.
    This function works by working out the difference of the two measurements, dm = m1 - m2.
        If the error on this difference is greater than the difference, i.e. the difference agrees within error with 0, then that means that the two values are close enough to say that they don't disagree within error, i.e. we can't say they are inconsistent.
        If the error is less than the difference, then they don't agree within error, i.e. we can say that they are inconsistent.
    The function also prints the number of sigma that they disagree by, (and a difference of less than 1 indicates agreement).
    
    Input
        m1, m2 - the two measurements, which are either ufloats or uarrays
    Outpit
        true, false - whether the two measurements agree within error
    """
    # If we are looking at two ufloats, not uarrays
    if not isinstance(m1, np.ndarray):
        dm_12 = abs(m1 - m2) # Calculates the absolute difference in the measurements
        num_sigma = dm_12.n/dm_12.s # Works out the number of standard deviations the difference is from 0
        # .n retrieves the nominal value of the ufloat
        # .s retrieves the error of the ufloat
        print(f'{num_sigma:.5}', "sigma difference")
        if num_sigma <= 1: # indicates agreement
            return True 
        else:
            return False
    else: # we are looking at uarrays, so carry out the same steps above per element of the array
        num_sigma_max = 0 # we say that the two arrays agree if and only if all the elements do, so we want the biggest sigma difference to be less than 1
        for i in range(len(m1)):
            dm_12 = abs(m1[i] - m2[i])
            num_sigma_i = dm_12.n/dm_12.s
            if num_sigma_i > num_sigma_max: # there is less agreement of this element
                num_sigma_max = num_sigma_i
        if num_sigma_max <= 1: # all elements agree with each other
            return True
        else:
            return False
    
def project_vector(vector, normal):
    """
    Obtains the projection of `vector' along `'normal'.
    Input
        vector - the vector whose component we are determining
        normal - the unit normal vector of the shock
    Output
        v_n - the projection of vector along normal
    """
    return vector @ normal

def transverse_vector(vector, normal):
    """
    Obtains the projection of `vector' along `'normal'.
    Input
        vector - the vector whose component we are determining
        normal - the normal vector of the shock
    Output
        v_t - the projection of vector along normal
    """
    return vector - (vector @ normal) * normal

def rankine_methods(B_up, B_dw, B_unit, V_up, V_dw, V_unit, rho_up, rho_dw, rho_unit, p_up, p_dw, avg_normals, avg_speeds, method_names):
    """
    The procedure loops through the normal vectors calculated by the Trotta code and for each determines whether the perpmeters satisfy the Rankine-Hugoniot equations we have listed.
    
    Input
        vec_B_up, etc. - our shock perameters
        avg_normals - the normal vectors obtained from the Trotta calculations
        avg_speeds - the shock speed calculated from the normal vector
        method_names - the name of each method associated with each vector, i.e. MC, MX1...
    """
    print("Rankine-Hugoniot Verification")
    
    for i in range(len(avg_normals)): # Loops through all the normal vectors
        normal = avg_normals[i]
        method = method_names[i]
        v_shock = avg_speeds[i]
        print("Method",method,"\nSpeed","{:fP}".format(v_shock),V_unit,"\n")
        
        rankine_analysis(B_up, B_dw, B_unit, V_up, V_dw, V_unit, rho_up, rho_dw, rho_unit, p_up, p_dw, normal, v_shock)

def rankine_analysis(B_up, B_dw, B_unit, V_up, V_dw, V_unit, rho_up, rho_dw, rho_unit, p_up, p_dw, normal, shock_speed=0):
    """
    A procedure that carries out the Rankine-Hugoniot analysis for our shock data
    In this procedure, _n refers to normal to the shock front, _t refers to transverse to the shock front
    
    Input
        vec_B_up, etc. - our shock perameters
        normal - the normal vector of the shock
        shock_speed - the shock speed calculated from the normal vector
    """

    # Equation 1 - [B_n] = 0?
    
    upstream_1, downstream_1 = rankine_1(B_up, B_dw, normal)
    
    data_func.print_data("Upstream B-field", upstream_1, B_unit)
    data_func.print_data("Downstream B-field", downstream_1, B_unit)
    data_func.print_data("Difference", downstream_1-upstream_1, B_unit)
    
    if within_error(upstream_1, downstream_1):
        print("Equation [B_n] = 0 is conserved.\n")
    else:
        print("Equation [B_n] = 0 is inconsistent.\n")

    # Equation 2 - [rho*v_n] = 0?
   
    upstream_2, downstream_2 = rankine_2(rho_up, rho_down, V_up, V_dw, normal, shock_speed)
    
    data_func.print_data("Upstream Mass Flow", upstream_2, V_unit+rho_unit)
    data_func.print_data("Downstream Mass Flow", downstream_2, V_unit+rho_unit)
    data_func.print_data("Difference", downstream_2-upstream_2, V_unit+rho_unit)
    
    if within_error(upstream_2, downstream_2):
        print("Equation [rho*V_n] = 0 is conserved.\n")
    else:
        print("Equation [rho*V_n] = 0 is inconsistent.\n")

    # Equation 3 - [B_n*V - v_n*B] = 0?
    
    upstream_3, downstream_3 = rankine_3(B_up, B_dw, V_up, V_dw, normal, shock_speed)
    
    data_func.print_data("Upstream Quantity", upstream_3, B_unit+V_unit)
    data_func.print_data("Downstream Quantity", downstream_3, B_unit+V_unit)
    data_func.print_data("Difference", downstream_3-upstream_3, B_unit+V_unit)
    
    if within_error(upstream_3, downstream_3):
        print("Equation [B_n*V - V_n*B] = 0 is conserved.\n")
    else:
        print("Equation [B_n*V - V_n*B] = 0 is inconsistent.\n")
        
    # Equation 4a - [rho*v_n^2+P+(B^2/2-B_n^2)/mu0] = 0?
    
    upstream_4a, downstream_4a = (rho_up, V_up, B_up, p_up, rho_dw, V_dw, B_dw, p_dw, normal, shock_speed)
    
    data_func.print_data("Upstream Pressure", upstream_4a, "nPa")
    data_func.print_data("Downstream Pressure", downstream_4a, "nPa")
    data_func.print_data("Difference", downstream_4a-upstream_4a, "nPa")
    
    if within_error(upstream_4a, downstream_4a):
        print("Equation [rho*v_n^2+P+(B^2/2-B_n^2)/mu0] = 0 is conserved.\n")
    else:
        print("Equation [rho*v_n^2+P+(B^2/2-B_n^2)/mu0] = 0 is inconsistent.\n")
        
    # Equation 4b - [rho*v_n*V_t-(B_nB_t)/mu0] = 0?

    upstream_4b, downstream_4b = rankine_4b(rho_up, rho_dw, V_up, V_dw, B_up, B_dw, normal, shock_speed)

    data_func.print_data("Upstream Pressure", upstream_4b, "nPa")
    data_func.print_data("Downstream Pressure", downstream_4b, "nPa")
    data_func.print_data("Difference", downstream_4b-upstream_4b, "nPa")
    
    if within_error(upstream_4b, downstream_4b):
        print("Equation [rho*v_n*V_t-(B_nB_t)/mu0] = 0 is conserved.\n")
    else:
        print("Equation [rho*v_n*V_t-(B_nB_t)/mu0] = 0 is inconsistent.\n")
    print("")
    
def rankine_1(B_up, B_dw, n):
    
    up_field = B_up @ n
    down_field = B_dw @ n
    
    return up_field, down_field
    
            
def rankine_2(rho_up, rho_down, V_up, V_dw, n, v_sh=0):
    
    V_n_up = V_up @ n - v_sh
    V_n_dw = V_dw @ n - v_sh
    
    up_flow = rho_up*V_n_up
    down_flow = rho_dw*V_n_dw
    
    return up_flow, down_flow    
    
def rankine_3(B_up, B_dw, V_up, V_dw, n, v_sh=0):
    
    
    # The bulk flow speeds transformed to be in the frame of the shock
    # By definition, the shock speed is in the direction of the normal
    # So we subtract the shock velocity from the bulk flow velocity vectors
    
    
    # Transform
    V_up = V_up - v_sh * n
    V_dw = V_dw - v_sh * n
    
    # Scalars
    B_n_up = B_up @ n
    B_n_dw = B_dw @ n
    
    V_n_up = V_up @ n
    V_n_dw = V_dw @ n
    
    up_field = B_n_up*V_up - V_n_up*B_up
    down_field = B_n_dw*V_dw - V_n_dw*B_dw
    
    return up_field, down_field
    
    
    
def rankine_4a(rho_up, V_up, B_up, p_up, rho_dw, V_dw, B_dw, p_dw, n, v_sh=0):
    
    # Transform
    V_up = V_up - v_sh * n
    V_dw = V_dw - v_sh * n
    
    # Scalars
    B_n_up = B_up @ n
    B_n_dw = B_dw @ n
    
    V_n_up = V_up @ n
    V_n_dw = V_dw @ n
    
    def rh4a(rho, v_n, p, B, b_n):
        return (con.m_p*rho*v_n**2)*10**(12)+p+(np.dot(B,B)/2-b_n**2)*10**(-18)/con.mu_0
    
    up_pressure = rh4a(rho_up,V_n_up,p_up,B_up,B_n_up)*10**9
    down_pressure = rh4a(rho_dw,V_n_dw,p_dw,B_dw,B_n_dw)*10**9
    
    return up_pressure, down_pressure
    
    
    
def rankine_4b(rho_up, rho_dw, V_up, V_dw, B_up, B_dw, n, v_sh=0):
    
    # Transform
    V_up = V_up - v_sh * n
    V_dw = V_dw - v_sh * n
    
    # Scalars
    B_n_up = B_up @ n
    B_n_dw = B_dw @ n
    
    V_n_up = V_up @ n
    V_n_dw = V_dw @ n
    
    # Vectors
    V_t_up = V_up - (V_up @ n) * n
    V_t_dw = V_dw - (V_dw @ n) * n
    
    B_t_up = B_up - (B_up @ n) * n
    B_t_dw = B_dw - (B_dw @ n) * n
    
    def rh4b(rho, v_n, V_t, b_n, B_t):
        return (con.m_p*rho*v_n*V_t)*10**(12)-(b_n*B_t)*10**(-18)/con.mu_0
    
    up_pressure = rh4b(rho_up,V_n_up,V_t_up,B_n_up,B_t_up)*10**9
    down_pressure = rh4b(rho_dw,V_n_dw,V_t_dw,B_n_dw,B_t_dw)*10**9
    
    return up_pressure, down_pressure