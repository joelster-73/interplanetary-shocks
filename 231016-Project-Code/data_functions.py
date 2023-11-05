#--------------- Importing key libraries and modules ---------------#
import time
from datetime import datetime, timedelta

import numpy as np
import scipy as spy
import math
from uncertainties import ufloat # for managing errors in the data
from uncertainties import unumpy
from uncertainties.umath import * # for mathematical operations, e.g. unumpy.sin(x)

import speasy as spz
import SerPyShock as SP

import import_ipynb


# Data trees

amda_tree = spz.inventories.tree.amda
cda_tree = spz.inventories.tree.cda
ssc_tree = spz.inventories.tree.ssc

"""## Data Retrieval"""

# Function to calculate the start and end time of the data taken
def find_time(date_shock, time_shock, time_period):
    """
    Author: Val Defayet
    Function to calculate the start and end time/date of the data to be analysed
    This function makes sure the dates and times used are in the correct datetime format
    This function reduces the complexity when trying to calculate times windows that span multiple days
    
    Input
        date_shock - the date when the shock is detected by the spacecraft
        time_shock - the time when the shock is detected by the spacecraft
        time_period - the time either side of when the shock is detected we are investigating
    Output
       start_date, end_date, time_shock - the date and time at the beginning and end of the window of investigation
    """
    if isinstance(time_shock, int):
        # Time is given in seconds
        hour_shock = int(time_shock // 3600)
        minute_shock = int((time_shock % 3600) // 60)
        second_shock = int(time_shock % 60)
    elif isinstance(time_shock, str):
        # Time is given in HH:MM:SS format
        hour_shock, minute_shock, second_shock = map(int, time_shock.split(':'))
    elif isinstance(time_shock, list):
        [hour_shock, minute_shock, second_shock] = [ int(x) for x in time_shock ]
    else:
        raise ValueError("Invalid time format. Please provide either seconds or HH:MM:SS.")

    [day_shock, month_shock, year_shock] = [ int(x) for x in date_shock ]

    # Define range to search the data and the time of the shock
    start_date = datetime(year_shock, month_shock, day_shock, hour_shock, minute_shock, second_shock) - timedelta(minutes=time_period)
    end_date = datetime(year_shock, month_shock, day_shock, hour_shock, minute_shock, second_shock) + timedelta(minutes=time_period)
    time_shock = datetime(year_shock, month_shock, day_shock, hour_shock, minute_shock, second_shock)

    dt = end_date - start_date

    return start_date, end_date, time_shock

# Function to get the data of the magnetic field
def get_Bfield(spacecraft, start_date, end_date):
    """
    Function to retrieve the magnetic field data using the speasy data library
    The function makes use of a dictionary of various spacecraft we are interested in
    This output is an array of all of the data recorded by the satellite within the time window passed in
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract
        start_date/end_date - the window of time in which we want to extract the data
    Output
        data.time - the time at which the spacecraft recorded the data
        data.values - the values of the data we are interested in
        data.unit - the unit of the data, e.g. nT, nanoteslas
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


# Function to get velocity on any satellite
def get_velocity(spacecraft, start_date, end_date):
    """
    Function to retrieve the bulk flow velocity data using the speasy data library
    The function makes use of a dictionary of various spacecraft we are interested in
    This output is an array of all of the data recorded by the satellite within the time window passed in
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract
        start_date/end_date - the window of time in which we want to extract the data
    Output
        data.time - the time at which the spacecraft recorded the data
        data.values - the values of the data we are interested in
        data.unit - the unit of the data, e.g. km/s
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

# Function to get density on any satellite
def get_density(spacecraft, start_date, end_date):
    """
    Function to retrieve the mass density data using the speasy data library
    The function makes use of a dictionary of various spacecraft we are interested in
    This output is an array of all of the data recorded by the satellite within the time window passed in
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract
        start_date/end_date - the window of time in which we want to extract the data
    Output
        data.time - the time at which the spacecraft recorded the data
        data.values - the values of the data we are interested in
        data.unit - the unit of the data, e.g. /cm3
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
        return data.time, data.values, data.unit
    raise Exception("Satellite not found")
    
# Function to get temperature on any satellite
def get_temperature(spacecraft, start_date, end_date):
    """
    Function to retrieve the temperature data using the speasy data library
    The function makes use of a dictionary of various spacecraft we are interested in
    This output is an array of all of the data recorded by the satellite within the time window passed in
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract
        start_date/end_date - the window of time in which we want to extract the data
    Output
        data.time - the time at which the spacecraft recorded the data
        data.values - the values of the data we are interested in
        data.unit - the unit of the data, e.g. km/s
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

# Function to get the coordinates of the satellites
def get_coordinate(spacecraft,time_shock):
    """
    Function to retrieve the coordinates using the speasy data library
    The function makes use of a dictionary of various spacecraft we are interested in
    This output is an array of all of the data recorded by the satellite within the time window passed in
    
    Input
        spacecraft - the name of the spacecraft whose data we wish to extract
        start_date/end_date - the window of time in which we want to extract the data
    Output
        time_orbit - the time at which the spacecraft recorded the data
        orbit - the coordinates of the satellite at the time we are interested
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
        data = spz.get_data(path, time_shock-timedelta(minutes=5), time_shock+timedelta(minutes=5))
        time_orbit = data.time
        orbit = data.values
        
        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit
    raise Exception("Satellite not found")

"""## Filtering"""

# Define function to filtered B within the averaging window
def filter_B(time_mag, Bx, By, Bz, time):
    """
    Function to filter the magnetic field data set such that data outside of the time window is removed/only looking at the data within the desired averaging window
    
    Input
        time_mag - the times at which the data is recorded
        Bx,By,Bz - the data we want to filter out
        time - the time window we want to filter the data within
    Output
        filtered data - the data filtered so that only the data within the time window is kept
    """    
    filtered_Bx = [] ; filtered_By = [] ; filtered_Bz = []

    for timestamp, bx, by, bz in zip(time_mag, Bx, By, Bz):
        truncated_timestamp = str(timestamp)[:26]  # Truncate to 6-digit fractional seconds
        timestamp_datetime = datetime.strptime(truncated_timestamp, '%Y-%m-%dT%H:%M:%S.%f')
        if time[0] <= timestamp_datetime <= time[1]:
            filtered_Bx.append(bx) ; filtered_By.append(by) ; filtered_Bz.append(bz)

    return np.array(filtered_Bx), np.array(filtered_By), np.array(filtered_Bz)

# Define function to filtered V winthin the averaging window
def filter_V(time_vel, Vx, Vy, Vz, time):
    """
    Function to filter the fulk flow speed data set such that data outside of the time window is removed/only looking at the data within the desired averaging window
    
    Input
        time_vel - the times at which the data is recorded
        Vx,Vy,Vz - the data we want to filter out
        time - the time window we want to filter the data within
    Output
        filtered data - the data filtered so that only the data within the time window is kept
    """    
    filtered_Vx = [] ; filtered_Vy = [] ; filtered_Vz = []

    for timestamp, vx, vy, vz in zip(time_vel, Vx, Vy, Vz):
        timestamp_str = str(timestamp).split('.')[0]  # Convert to string and remove fractional seconds
        timestamp_datetime = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
        if time[0] <= timestamp_datetime <= time[1]:
            filtered_Vx.append(vx) ; filtered_Vy.append(vy) ; filtered_Vz.append(vz)

    return np.array(filtered_Vx), np.array(filtered_Vy), np.array(filtered_Vz)

# Define function to filtered N winthin the averaging window
def filter_N(time_rho, Nx, time):
    """
    Function to filter the density data set such that data outside of the time window is removed/only looking at the data within the desired averaging window
    
    Input
        time_rho - the times at which the data is recorded
        Nx - the data we want to filter out
        time - the time window we want to filter the data within
    Output
        filtered data - the data filtered so that only the data within the time window is kept
    """    
    # Filter Nx values within the desired time range
    filtered_Nx = []

    for timestamp, nx in zip(time_rho, Nx):
        timestamp_str2 = str(timestamp).split('.')[0]  # Convert to string and remove fractional seconds
        timestamp_datetime = datetime.strptime(timestamp_str2, '%Y-%m-%dT%H:%M:%S')
        if time[0] <= timestamp_datetime <= time[1]:
            filtered_Nx.append(nx)

    return np.array(filtered_Nx)

# Define function to filtered P winthin the averaging window
def filter_P(time_pres, Px, time):
    """
    Function to filter the density data set such that data outside of the time window is removed/only looking at the data within the desired averaging window
    
    Input
        time_pres - the times at which the data is recorded
        Nx - the data we want to filter out
        time - the time window we want to filter the data within
    Output
        filtered data - the data filtered so that only the data within the time window is kept
    """    
    # Filter Px values within the desired time range
    filtered_Px = []

    for timestamp, px in zip(time_pres, Px):
        timestamp_str2 = str(timestamp).split('.')[0]  # Convert to string and remove fractional seconds
        timestamp_datetime = datetime.strptime(timestamp_str2, '%Y-%m-%dT%H:%M:%S')
        if time[0] <= timestamp_datetime <= time[1]:
            filtered_Px.append(px)

    return np.array(filtered_Px)

"""
Printing Data
"""

def nom_vals(uarr):
    """
    Renaming the function in the unumpy module to obtain the mean vaules in an array of ufloats
    
    Input
        uarr - input of array of ufloats
    Output
        array - array of the mean values of the ufloats
    """
    return unumpy.nominal_values(uarr)

def string_uarray(uarr):
    """
    Function to convert an array of ufloats into a readable string
    
    Input
        uarr - input of array of ufloats
    Output
        string - string in the format of '[means] ± [errors]'
    """
    mean_str = "[" ; err_str = "["
    for ele in uarr:
        ele = "{:f}".format(ele)
        mean_str += ele.split("+/-")[0].strip() + ", "
        err_str += ele.split("+/-")[1].strip() + ", "
    mean_str = mean_str[:-2] + "]" ; err_str = err_str[:-2] + "]"
    return mean_str + " ± " + err_str
    
def print_data(label, data, unit="", method=None):
    """
    Procedure to print out the data of interest in a nicely presented format
    
    Input
        label - a description of the data
        data - the actual data being looked at
        unit - the unit of th edata
        method - the normal method used      
    """
    if method is not None:
        print("Method",method)
    if isinstance(data, np.ndarray):
        print(label+":",string_uarray(data),str(unit))
    else:
        print(label+":","{:fP}".format(data),str(unit))