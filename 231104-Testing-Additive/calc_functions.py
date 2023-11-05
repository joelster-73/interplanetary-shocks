""" Importing key libraries and modules """
import time
from datetime import datetime, timedelta

import numpy as np
import scipy.constants as con
import math
from scipy.interpolate import interp1d

from uncertainties import ufloat # for managing errors in the data
from uncertainties import unumpy
from uncertainties.umath import * # for mathematical operations, e.g. unumpy.sin(x)
 
import speasy as spz
import SerPyShock as SP
from data_functions import *

import import_ipynb

# Defining the functions used in handling the uncertainties in the data 
def mean_std(data_range, the_axis=None):
    """
    Function to return a ufloat object, storing the average and standard deviation
    For the data_range provided, usually a filtered window
    the_axis is the axis along which the mean and std calculations are taken
        axis = None flattens the array to get mean of every element
        axis = 0 calculates for each column, (i.e. for each 1st, 2nd etc. element of each subarray)
        axis = 1 calculates for each row, (i.e. the mean of the first subarray, second etc.)
    
    Input
        data_range - the data of which we want to take the mean and standard deviation
        the_axis - the axis along which we want to take the mean
    Output
        ufloat/uarray - the array or singular mean and error
    """
    if the_axis == 0:
        return np.array([ufloat(np.nanmean(data_range[:,i]), np.std(data_range[:,i])) for i in range(len(data_range[0]))])
    elif the_axis == 1:
        return np.array([ufloat(np.nanmean(data_range[i]), np.std(data_range[i])) for i in range(len(data_range))])
    
    return ufloat(np.nanmean(data_range), np.std(data_range)) # Doesn't work with ufloats and axis=0,1

def nom_vals(uarr):
    """
    Renaming the function in the unumpy module to obtain the mean vaules in an array of ufloats.
    
    Input
        uarr - input of array of ufloats
    Output
        array - array of the mean values of the ufloats
    """
    return unumpy.nominal_values(uarr)

# Function to calculate magnitude
def magnitude(vector):
    """
    Function to calculate the magnitude of a vector
    np.linalg.norm doesn't work with uarrays
    
    Input
        vector - the vector we want to calculate the magnitude of
    Output
        magnitude - the magnitude of the vector
    """
    return (sum(element**2 for element in vector))**(0.5)

def pressure_temp(rho, temp):
    """
    Function to calculate the thermal pressure from the number density and temperature
        P = nkT
    We pass the temperature in in eV, and density in cm-3, so
        p = rho(m-3) * temp(J)
          = rho * 10**6 * temp(eV) * e
    Return pressure in Pa
    
    Input
        rho - the number density in cm-3
        temp - the temperature in eV
    Output
        pres - the pressure in Pa
    """
    return (rho*10**6)*(temp*con.e)

def pressure_vth(m_rho, vth):
    """
    Function to calculate the thermal pressure from the mass density and thermal velocity
    vth = sqrt(2kT/m) => kT = mv^2/2
                      => P = nmv^2/2
                      => P = rho*(vth^2)/2
    We pass the velocity in in km/s, and density in kgcm-3, so
        p = rho*(vth^2)/2
          = rho*10**6 * ((vth*10**3)^2)/2
    Return pressure in Pa
    
    Input
        m_rho - the mass density in kg cm-3
        vth - the thermal velocity in km s-1
    Output
        pres - the pressure in Pa
    """
    return (m_rho*10**6)*((vth*10**3)**2)/2


def extend_data(B, V, rho, pres):
    """
    Function to extend the bulk flow speed, density, and pressure data to match the same precision as the magnetic field data.
    The magnetic field data contains many more data values. 
    So, when calculating, we need the other parameters to match the same time precision.
    We do this through interpolation.
    
    Input
        B - the magnetic field density
        V - the bulk flow velocity
        rho - the number density
        pres - the pressure
    Output
        extended data - the data containing the same number of values as B
    """
    # Determining the extended velocity vector to be used in the Trotta method calculations
    data = V

    # Number of indexes to extend
    num_indexes = np.shape(B)[0]

    # Generate an array of indices
    indices = np.arange(data.shape[0])

    # Interpolate each column separately
    interpolated_data_V = np.empty((num_indexes, data.shape[1]))
    for col in range(data.shape[1]):
        interpolator = interp1d(indices, data[:, col], kind="cubic")
        interpolated_data_V[:, col] = interpolator(np.linspace(0, indices[-1], num_indexes))

    # Make an array with the data
    indices_rho = np.linspace(0, len(rho) - 1, num=len(rho))
    new_indices_rho = np.linspace(0, len(rho) - 1, num=np.shape(B)[0])

    # Flattens the arrays
    original_data_flat = np.ravel(rho)
    indices_flat = np.ravel(indices_rho)

    # Perfors linear interpolation
    extended_rho = np.interp(new_indices_rho, indices_flat, original_data_flat)
    
    # Make an array with the data
    indices_pres = np.linspace(0, len(pres) - 1, num=len(pres))
    new_indices_pres = np.linspace(0, len(pres) - 1, num=np.shape(B)[0])

    # Flattens the arrays
    original_data_flat = np.ravel(pres)
    indices_flat = np.ravel(indices_pres)

    # Perfors linear interpolation
    extended_pres = np.interp(new_indices_pres, indices_flat, original_data_flat)
    
    return interpolated_data_V, extended_rho, extended_pres

# Calculate the velocity with averaging windows
def SP_shock_speed(times, B, V, rho, time_shock, up_shk, dw_shk, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad, method="MX3", V_unit="km/s"):
    """
    Function to calculate the speed of a shock using the retrieved shock parameters and multiple averaging windows as defined.
    We are using a particular method to calculate the normal vector, MX3 by default.
    The function uses the SP methods as defined by Trotta.
    
    Input
        times - the times at which the data are recorded
        B, V, rho - the plasma parameters used to calculate the shock normal and speed
        window parameters - the parameters for the minimum and maximum window widths for the Trotta routines
        method - the method used to calculate the normal vector
    Output
        normal, angle, speed - the parameters of the shock
    """    
    # Calculate the normal with multiple averaging windows
    n, theta_Bn, rB, ex = SP.MX_stats(times, B, times, V, time_shock, up_shk, dw_shk, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad, "GSE")
    
    # Creating a dictionary to store the options of satellites and the calculated normal vector and shock angle
    vars_dict = {
        "MC": (n.MC, theta_Bn.MC),
        "MX1": (n.MX1, theta_Bn.MX1),
        "MX2": (n.MX2, theta_Bn.MX2),
        "MX3": (n.MX3, theta_Bn.MX3)
    }
    # Retrives the vector and angle based on the spacecraft input
    params = vars_dict.get(method)
    if params is not None:
        (sp_normals, sp_angles) = params
    else:
        raise Exception("Method not found")
    
    # Calculates the average normal
    avg_normal = mean_std(sp_normals, 0)
    avg_normal_unit = avg_normal / magnitude(avg_normal)
    
    # Calculates the average angle
    avg_angle = mean_std(sp_angles)
    
    # Uses the calculated normal vector to obtain the speed of the shock
    window_speeds, ex = SP.Vsh_stats(nom_vals(avg_normal_unit), times, V, rho, time_shock, up_shk, dw_shk, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad)
    avg_speed = mean_std(window_speeds)

    return avg_normal_unit, avg_angle, avg_speed
