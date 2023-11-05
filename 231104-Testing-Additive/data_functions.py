""" Importing key libraries and modules """
import time
from datetime import datetime, timedelta

import numpy as np
import scipy.constants as con
import math

from uncertainties import ufloat # for managing errors in the data
from uncertainties import unumpy
from uncertainties.umath import * # for mathematical operations, e.g. unumpy.sin(x)

import speasy as spz
import SerPyShock as SP
from calc_functions import *

# Defining speasy databases for the spacecraft data
amda_tree = spz.inventories.tree.amda
cda_tree = spz.inventories.tree.cda
ssc_tree = spz.inventories.tree.ssc

"""Data Retrieval"""

# Function to get the coordinates of the satellites
def get_coordinate(spacecraft,time_shock):
    """
    Function to retrieve the coordinates using the speasy data library.
    The function makes use of a dictionary of various spacecraft we are interested in.
    This output is an array of all of the data recorded by the satellite within the time window passed in.
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract.
        start_date/end_date - the window of time in which we want to extract the data.
    Output
        time_orbit - the time at which the spacecraft recorded the data.
        orbit - the coordinates of the satellite at the time we are interested.
    """    
    cluster = amda_tree.Parameters.Cluster
    themis = amda_tree.Parameters.THEMIS
    
    # Creating a dictionary to store the options of satellites and the retrieved data
    coord_dict = {
        "C1": cluster.Cluster_1.Ephemeris.clust1_orb_all.c1_xyz_gse,
        "C3": cluster.Cluster_3.Ephemeris.clust3_orb_all.c3_xyz_gse,
        #"C4": cluster.Cluster_4.Ephemeris.clust4_orb_all.c4_xyz_gse,
        "THA": themis.THEMIS_A.Ephemeris.tha_orb_all.tha_xyz,
        "THB": themis.THEMIS_B.Ephemeris.thb_orb_all.thb_xyz,
        "THC": themis.THEMIS_C.Ephemeris.thc_orb_all.thc_xyz,
        "THD": themis.THEMIS_D.Ephemeris.thd_orb_all.thd_xyz,
        "THE": themis.THEMIS_E.Ephemeris.the_orb_all.the_xyz,
        "WIND": amda_tree.Parameters.Wind.Ephemeris.wnd_orb_all.wnd_xyz_gse
    }
    # Retrives the path based on the spacecraft input
    path = coord_dict.get(spacecraft)
    if path is not None:
        data = spz.get_data(path, time_shock-timedelta(minutes=10), time_shock+timedelta(minutes=10))
        return data.time, data.values
    raise Exception("Satellite not found")
    
def satellite_info(spacecraft, time_shock):
    """
    Procedure to print the coordinates of the spacecraft at the designated time of the shock.
    Input
        spacecraft - the satellite of interest
        time_shock - the time of the shock
    """
    time, coord = get_coordinate(spacecraft, time_shock)
    index = find_nearest(time, np.datetime64(time_shock)) # finds the closest time in the list of times to the shock time
    print("The time is:",to_datetime(time[index]))
    print("The satellite coordinates are:",coord[index])

# Function to get magnetic field at any satellite
def get_Bfield(spacecraft, start_date, end_date):
    """
    Function to retrieve the magnetic field data using the speasy data library.
    The function makes use of a dictionary of various spacecraft we are interested in.
    This output is an array of all of the data recorded by the satellite within the time window passed in.
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract.
        start_date/end_date - the window of time in which we want to extract the data.
    Output
        data.time - the time at which the spacecraft recorded the data.
        data.values - the values of the data we are interested in.
        data.unit - the unit of the data.
    """
    cluster = amda_tree.Parameters.Cluster
    themis = amda_tree.Parameters.THEMIS
    
    # Creating a dictionary to store the options of satellites and the data paths 
    mag_dict = {
        "C1": cluster.Cluster_1.FGM.clust1_fgm_high.c1_b_5vps,
        "C3": cluster.Cluster_3.FGM.clust3_fgm_high.c3_b_5vps,
        #"C4": cluster.Cluster_4.FGM.clust4_fgm_high.c4_b_5vps,
        "THA": themis.THEMIS_A.FGM.tha_fgm_s.tha_bs,
        "THB": themis.THEMIS_B.FGM.thb_fgm_s.thb_bs,
        "THC": themis.THEMIS_C.FGM.thc_fgm_s.thc_bs,
        "THD": themis.THEMIS_D.FGM.thd_fgm_s.thd_bs,
        "THE": themis.THEMIS_E.FGM.the_fgm_s.the_bs,
        "WIND": amda_tree.Parameters.Wind.MFI.wnd_mfi_high.wnd_bh
    }
    # Retrives the path based on the spacecraft input
    path = mag_dict.get(spacecraft)
    if path is not None:
        data = spz.get_data(path, start_date, end_date)
        return data.time, data.values, data.unit
    raise Exception("Satellite not found")

# Function to get velocity at any satellite
def get_velocity(spacecraft, start_date, end_date):
    """
    Function to retrieve the bulk flow data using the speasy data library.
    The function makes use of a dictionary of various spacecraft we are interested in.
    This output is an array of all of the data recorded by the satellite within the time window passed in.
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract.
        start_date/end_date - the window of time in which we want to extract the data.
    Output
        data.time - the time at which the spacecraft recorded the data.
        data.values - the values of the data we are interested in.
        data.unit - the unit of the data.
    """ 
    cluster = amda_tree.Parameters.Cluster
    themis = amda_tree.Parameters.THEMIS
    
    # Creating a dictionary to store the options of satellites and the retrieved data
    vel_dict = {
        "C1": cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_v,
        "C3": cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_v,
        #"C4": cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_v,
        "THA": themis.THEMIS_A.ESA.tha_peim_all.tha_v_peim,
        "THB": themis.THEMIS_B.ESA.thb_peim_all.thb_v_peim,
        "THC": themis.THEMIS_C.ESA.thc_peim_all.thc_v_peim,
        "THD": themis.THEMIS_D.ESA.thd_peim_all.thd_v_peim,
        "THE": themis.THEMIS_E.ESA.the_peim_all.the_v_peim,
        "WIND": amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_v
    }
    # Retrives the path based on the spacecraft input
    path = vel_dict.get(spacecraft)
    if path is not None:
        data = spz.get_data(path, start_date, end_date)
        return data.time, data.values, data.unit
    raise Exception("Satellite not found")

# Function to get density at any satellite
def get_density(spacecraft, start_date, end_date):
    """
    Function to retrieve the number density data using the speasy data library.
    The function makes use of a dictionary of various spacecraft we are interested in.
    This output is an array of all of the data recorded by the satellite within the time window passed in.
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract.
        start_date/end_date - the window of time in which we want to extract the data.
    Output
        data.time - the time at which the spacecraft recorded the data.
        data.values - the values of the data we are interested in.
        data.unit - the unit of the data.
    """  
    cluster = amda_tree.Parameters.Cluster
    themis = amda_tree.Parameters.THEMIS
    
    # Creating a dictionary to store the options of satellites and the retrieved data
    dens_dict = {
        "C1": cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_dens,
        "C3": cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_dens,
        #"C4": cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_dens,
        "THA": themis.THEMIS_A.ESA.tha_peim_all.tha_n_peim,
        "THB": themis.THEMIS_B.ESA.thb_peim_all.thb_n_peim,
        "THC": themis.THEMIS_C.ESA.thc_peim_all.thc_n_peim,
        "THD": themis.THEMIS_D.ESA.thd_peim_all.thd_n_peim,
        "THE": themis.THEMIS_E.ESA.the_peim_all.the_n_peim,
        "WIND": amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_n
    }
    # Retrives the path based on the spacecraft input
    path = dens_dict.get(spacecraft)
    if path is not None:
        data = spz.get_data(path, start_date, end_date)
        vals = data.values
        return data.time, vals[:,0], data.unit
    raise Exception("Satellite not found")
    
# Function to get temperature at any satellite
def get_temperature(spacecraft, start_date, end_date):
    """
    Function to retrieve the temperature data using the speasy data library.
    The function makes use of a dictionary of various spacecraft we are interested in.
    This output is an array of all of the data recorded by the satellite within the time window passed in.
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract.
        start_date/end_date - the window of time in which we want to extract the data.
    Output
        data.time - the time at which the spacecraft recorded the data.
        data.values - the values of the data we are interested in.
        data.unit - the unit of the data.
    """ 
    cluster = amda_tree.Parameters.Cluster
    themis = amda_tree.Parameters.THEMIS
    
    # Creating a dictionary to store the options of satellites and the retrieved data
    temp_dict = {
        "C1": cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_t,
        "C3": cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_t,
        #"C4": cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_t,
        "THA": themis.THEMIS_A.ESA.tha_peim_all.tha_t_peim,
        "THB": themis.THEMIS_B.ESA.thb_peim_all.thb_t_peim,
        "THC": themis.THEMIS_C.ESA.thc_peim_all.thc_t_peim,
        "THD": themis.THEMIS_D.ESA.thd_peim_all.thd_t_peim,
        "THE": themis.THEMIS_E.ESA.the_peim_all.the_t_peim,
        "WIND": amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_vth
    }
    # Retrives the path based on the spacecraft input
    path = temp_dict.get(spacecraft)
    if path is not None:
        data = spz.get_data(path, start_date, end_date)
        return data.time, data.values, data.unit
    raise Exception("Satellite not found")

# Function to get pressure at any satellite
def get_pressure(name_spacecraft, start_date, end_date, rho):
    """
    Function to retrieve the pressure data using the speasy data library.
    The function makes use of a dictionary of various spacecraft we are interested in.
    This output is an array of all of the data recorded by the satellite within the time window passed in.
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract.
        start_date/end_date - the window of time in which we want to extract the data.
    Output
        data.time - the time at which the spacecraft recorded the data.
        data.values - the values of the data we are interested in.
        data.unit - the unit of the data.
    """ 
    # Retrieves temperature data
    time_temp, temp, temp_unit = get_temperature(name_spacecraft, start_date, end_date)
    
    # Reshapes the data
    Temp = np.array(temp[:,0])
    
    # Determining which equation we need to use to calculate the pressure
    if name_spacecraft=="WIND":
        Pres = pressure_vth(rho*con.m_p, Temp)
    else:
        Pres = pressure_temp(rho, Temp)

    return time_temp, np.array(Pres), "Pa"
    
"""Filtering"""

# Function to filter variable data within a time window
def filter_var(time_var, Var, window, dim):
    """
    Function to filter a list of data so we only return the data within the time window.
    This makes use of the SerPyShock functions developed by Trotta.
    
    Input
        time_var - the list of times at which the data is recorded
        Var - the list of data we are wanting to filter
        window - the time window we want to filter the data within, (i.e. we only want the data recorded within this window of time)
        dim - the number of dimensions of the data, (i.e. dim=1 for a list of scalars, and dim=3 for a list of vectors)
    Output
        data_avg - the average of the data within the time window
    """ 
    # The indices of the list corresponding to the time window
    start, end = SP.get_time_indices(np.datetime64(window[0]), np.datetime64(window[1]), time_var)
    # The times and data within the filtered time window
    times, data = SP.select_subS(Var, time_var, start, end, dim)
    # If we are looking at vectors rather than scalars
    if dim > 1:     
        data_avg = mean_std(data,0) 
    else:
        data_avg = mean_std(data)
    return data_avg

def filter_up_dw(time_var, Var, window_up, window_dw, dim=3):
    """
    Function to filter a list of data in the upstream and downstream time windows.
    
    Input
        time_var - the list of times at which the data is recorded
        Var - the list of data we are wanting to filter
        window_up, window_dw - the time window we want to filter the data within, (i.e. we only want the data recorded within this window of time)
        dim - the number of dimensions of the data, (i.e. dim=1 for a list of scalars, and dim=3 for a list of vectors)
    Output
        avg_up, avg_dw - the average of the data within the upstream, downstream time windows
    """ 
    avg_up = filter_var(time_var, Var, window_up, dim)
    avg_dw = filter_var(time_var, Var, window_dw, dim)
    
    return avg_up, avg_dw
    
def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object.
    
    Input
      date - a np.datetime64 object
    Output
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64("1970-01-01T00:00:00"))
                 / np.timedelta64(1, "s"))
    return datetime.utcfromtimestamp(timestamp)
    
def find_nearest(array, value):
    """
    Function to retrieve the index of the closest item in array to value.
    
    Input
        array - the array we are looking at
        value - the value we are looking for
    Output
        index - the index of the closest value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

""" Printing Data """

def vector_uarray(uarr):
    """
    Function to convert an array of ufloats into a readable string.
    
    Input
        uarr - input of array of ufloats
    Output
        string - string in the format of '[means] ± [errors]'
    """
    mean_arr = [] ; err_arr = []
    for ele in uarr:
        ele = "{:f}".format(ele)
        mean_arr.append(float(ele.split("+/-")[0].strip()))
        err_arr.append(float(ele.split("+/-")[1].strip()))
    return mean_arr, err_arr

def print_data(label, data, unit="", method=None):
    """
    Procedure to print out the data of interest in a nicely presented format.
    
    Input
        label - a description of the data
        data - the actual data being looked at
        unit - the unit of th edata
        method - the normal method used      
    """
    if method is not None:
        print("Method",method)
    if isinstance(data, np.ndarray):
        mean, err = vector_uarray(data)
        data_string = str(mean) + " ± " + str(err)
        print(label+":",data_string,str(unit))
    else:
        print(label+":","{:fP}".format(data),str(unit))

def print_avg(var_avg, var_name, var_unit, var_dim=3):
    """
    Procedure to print the average of the variable over the time window.
    If the variable is a vector, then the magnitude of the vector is also printed.
    
    Input
        var_avg - the average of the variable over the time window
        var_name - the name of the variable, for printing purposes
        var_unit - the unit of the variable
        var_dim - the number of dimensions of the variable
    """ 
    print_data(var_name, var_avg, var_unit)
    if var_dim > 1:
        print_data("Magnitude", magnitude(var_avg), var_unit)
        
# Function to plot characteristics on the figures
def plot_line(up_windows, dw_windows, up_shk, dw_shk, time_shock, ax):
    """
    Procedure to plot certain characteristics on our figures of interest
    
    Input
        up_windows, dw_windows - the minimum and maximum upstream, downstream window ranges
        up_shk, dw_shk - where the upstream, downstream regions begin
        time_shock - the time of the shock
        ax - the axis on which we are plotting
    """
    # Dark colour for the minimum region of averaging
    ax.axvspan(up_shk,up_windows[0], facecolor = "green", alpha = 0.9)
    ax.axvspan(dw_shk,dw_windows[0], facecolor = "yellow", alpha = 0.9)

    # Light colour for the maximum region of averaging
    ax.axvspan(up_windows[0],up_windows[1], facecolor = "green", alpha = 0.3)
    ax.axvspan(dw_windows[0],dw_windows[1], facecolor = "yellow", alpha = 0.3)
    
    # Vertical line for the time of the shock
    ax.axvline(x = time_shock, color = "red", ls="--", lw=2)