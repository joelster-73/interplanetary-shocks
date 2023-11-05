#--------------- Importing key libraries and modules ---------------#
import time
from datetime import datetime, timedelta

import numpy as np
import scipy as spy
import scipy.constants as con
import math
from uncertainties import ufloat # for managing errors in the data
from uncertainties import unumpy
from uncertainties.umath import * # for mathematical operations, e.g. unumpy.sin(x)

import speasy as spz
import SerPyShock as SP
import data_functions as data_func

import import_ipynb

"""## Analysis"""

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

def angle_vectors(vec1, vec2):
    """
    A function to calculate the angle between two vectors.
    The numpy.arccos function calculates the output in radians, so we convert to degrees
    np.arccos doesn't work with uarrays, so we use unumpy.arccos
    
    Input
        vec1, vec2 - two vectors we are looking at
    Output
        angle - the angle between the two vectors in degrees
    """
    v1 = vec1 / magnitude(vec1) ; v2 = vec2 / magnitude(vec2)
    angle_rad = unumpy.arccos(np.dot(v1,v2))
    angle = angle_rad * 180/np.pi
    if angle > 90:
        angle = 180 - angle
    return angle

"""## Normal Methods"""        

def normal_method(Bup, Bdw, dB, dV, method):
    """
    Procedure to calculate the normal vector to the shock using the 'method' chosen
    The angle between the upstream magnetic field and the normal is also calculated
    Both these values are appended to their appropiate arrays.
    
    Input
        Bdw, Bup - the downstream and upstream magnetic fields
        dB - the difference in the downstream and upstream fields, i.e. dB=Bdw-Bup
        dV - the difference in the downstream and upstream bulk flow speeds, i.e. dV = Vdw - Vup
    """
    normal = method(Bup, Bdw, dB, dV)
    if(normal[0] > 0): # Adjust the normal vector to point upstream, i.e. in the -x direction
        normal = -normal
    theta_Bup_n = angle_vectors(Bup, normal)
    return normal, theta_Bup_n

def normal_MC(Bup, Bdw, dB, dV):
    """
    Procedure to calculate the normal vector of the shock using the MC method
    n = (Bdw x Bup) x dB
    
    Input
        Bdw, Bup - the downstream and upstream magnetic fields
        dB - the difference in the downstream and upstream fields, i.e. dB=Bdw-Bup
        dV - the difference in the downstream and upstream bulk flow speeds, i.e. dV = Vdw - Vup
    """
    Bdw_x_Bup = np.cross(Bdw, Bup)
    n_MC = np.cross(Bdw_x_Bup, dB)
    unit_n_MC = n_MC / magnitude(n_MC)
    return unit_n_MC

def normal_MX1(Bup, Bdw, dB, dV):
    """
    Procedure to calculate the normal vector of the shock using the MX1 method
    n = (Bup x dV) x dB
    
    Input
        Bdw, Bup - the downstream and upstream magnetic fields
        dB - the difference in the downstream and upstream fields, i.e. dB=Bdw-Bup
        dV - the difference in the downstream and upstream bulk flow speeds, i.e. dV = Vdw - Vup
    """
    Bup_x_dV = np.cross(Bup, dV)
    n_MX1 = np.cross(Bup_x_dV, dB)
    unit_n_MX1 = n_MX1 / magnitude(n_MX1)
    return unit_n_MX1

def normal_MX2(Bup, Bdw, dB, dV):
    """
    Procedure to calculate the normal vector of the shock using the MX2 method
    n = (Bdw x dV) x dB
    
    Input
        Bdw, Bup - the downstream and upstream magnetic fields
        dB - the difference in the downstream and upstream fields, i.e. dB=Bdw-Bup
        dV - the difference in the downstream and upstream bulk flow speeds, i.e. dV = Vdw - Vup
    """
    Bdw_x_dV = np.cross(Bdw, dV)
    n_MX2 = np.cross(Bdw_x_dV, dB)
    unit_n_MX2 = n_MX2 / magnitude(n_MX2)
    return unit_n_MX2

def normal_MX3(Bup, Bdw, dB, dV):
    """
    Procedure to calculate the normal vector of the shock using the MX3 method
    n = (dV x dV) x dB
    
    Input
        Bdw, Bup - the downstream and upstream magnetic fields
        dB - the difference in the downstream and upstream fields, i.e. dB=Bdw-Bup
        dV - the difference in the downstream and upstream bulk flow speeds, i.e. dV = Vdw - Vup
    """
    dB_x_dV = np.cross(dB, dV)
    n_MX3 = np.cross(dB_x_dV, dB)
    unit_n_MX3 = n_MX3 / magnitude(n_MX3)
    return unit_n_MX3

def normal_VC(Bup, Bdw, dB, dV):
    """
    Procedure to calculate the normal vector of the shock using the MC method
    n = dV
    
    Input
        Bdw, Bup - the downstream and upstream magnetic fields
        dB - the difference in the downstream and upstream fields, i.e. dB=Bdw-Bup
        dV - the difference in the downstream and upstream bulk flow speeds, i.e. dV = Vdw - Vup
    """
    unit_n_VC = dV / magnitude(dV)
    return unit_n_VC

def pressure_temp(rho, temp):
    """
    Function to calculate the thermal pressure from the number density and temperature
        P = nkT
    We pass the temperature in in eV, and density in cm-3, so
        p = rho(m-3) * temp(J)
          = rho * 10**6 * temp(eV) * e
    Return pressure in Pa
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
    """
    return (m_rho*10**6)*((vth*10**3)**2)/2