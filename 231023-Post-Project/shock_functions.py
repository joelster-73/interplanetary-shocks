#--------------- Importing key libraries and modules ---------------#
import time
from datetime import datetime, timedelta

import numpy as np
import scipy as spy
import scipy.constants as con
import math
from scipy.interpolate import interp1d
from uncertainties import ufloat # for managing errors in the data
from uncertainties import unumpy
from uncertainties.umath import * # for mathematical operations, e.g. unumpy.sin(x)

import speasy as spz
import SerPyShock as SP
import data_functions as data_func
import calc_functions as calc_func

def one_window_mag(name_spacecraft, start_date, end_date, avg_windows):
    #--------------- Retrieves magnetic field data ---------------#
    time_mag, B, B_unit = data_func.get_Bfield(name_spacecraft, start_date, end_date)

    Bx = np.array([b[0] for b in B])
    By = np.array([b[1] for b in B])
    Bz = np.array([b[2] for b in B])

    # Defining the averaging windows for B_up, B_dw and filtering the magnetic field
    # By filtering, we are slicing the data to look at a smaller range of data
    filtered_Bx_up, filtered_By_up, filtered_Bz_up = data_func.filter_B(time_mag, Bx, By, Bz, avg_windows[0])
    filtered_Bx_dw, filtered_By_dw, filtered_Bz_dw = data_func.filter_B(time_mag, Bx, By, Bz, avg_windows[1])

    # Ufloat objects for the average of the upstream and downstream magnetic field components, and their associated uncertainty
    Bx_up = calc_func.mean_std(filtered_Bx_up) ; By_up = calc_func.mean_std(filtered_By_up) ; Bz_up = calc_func.mean_std(filtered_Bz_up)
    Bx_dw = calc_func.mean_std(filtered_Bx_dw) ; By_dw = calc_func.mean_std(filtered_By_dw) ; Bz_dw = calc_func.mean_std(filtered_Bz_dw)

    # Creates a vector B for the averaging window
    vec_B_up = np.array([Bx_up, By_up, Bz_up])
    vec_B_dw = np.array([Bx_dw, By_dw, Bz_dw])

    #print("There are",len(B),"data values of the magnetic field in this date range.")
    #print("There are",len(filtered_Bx_up),"data values of the sliced magnetic field in this time range.\n")

    data_func.print_data("Upstream Magnetic Field", vec_B_up, B_unit)
    data_func.print_data("Downstream Magnetic Field", vec_B_dw, B_unit)
    print("")
    
    return time_mag, B, B_unit, vec_B_up, vec_B_dw

def one_window_vel(name_spacecraft, start_date, end_date, avg_windows):
    #--------------- Retrieves velocity data ---------------#
    time_vel, V, V_unit = data_func.get_velocity(name_spacecraft, start_date, end_date)

    Vx = np.array([v[0] for v in V])
    Vy = np.array([v[1] for v in V])
    Vz = np.array([v[2] for v in V])

    # Defining the averaging windows for V_up, V_dw and filtering the velocity
    filtered_Vx_up, filtered_Vy_up, filtered_Vz_up = data_func.filter_V(time_vel, Vx, Vy, Vz, avg_windows[0])
    filtered_Vx_dw, filtered_Vy_dw, filtered_Vz_dw = data_func.filter_V(time_vel, Vx, Vy, Vz, avg_windows[1])

    # Ufloat objects for the average of the upstream and downstream velocity components, and their associated uncertainty
    Vx_up = calc_func.mean_std(filtered_Vx_up) ; Vy_up = calc_func.mean_std(filtered_Vy_up) ; Vz_up = calc_func.mean_std(filtered_Vz_up)
    Vx_dw = calc_func.mean_std(filtered_Vx_dw) ; Vy_dw = calc_func.mean_std(filtered_Vy_dw) ; Vz_dw = calc_func.mean_std(filtered_Vz_dw)

    # create a vector V for a defined range
    vec_V_up = np.array([Vx_up, Vy_up, Vz_up])
    vec_V_dw = np.array([Vx_dw, Vy_dw, Vz_dw])

    #print("There are",len(V),"data values of the bulk flow velocity in this date range.")
    #print("There are",len(filtered_Vx_up),"data values of the sliced bulk flow velocity in this time range.\n")

    data_func.print_data("Upstream Velocity", vec_V_up, V_unit)
    data_func.print_data("Downstream Velocity", vec_V_dw, V_unit)
    print("")
    
    return time_vel, V, V_unit, vec_V_up, vec_V_dw

def one_window_den(name_spacecraft, start_date, end_date, avg_windows):
    #--------------- Retrieves density data ---------------#
    time_rho, rho, rho_unit = data_func.get_density(name_spacecraft, start_date, end_date)

    Rho = np.array([Density[0] for Density in rho])

    # Defining the averaging windows for n_up, n_dw
    filtered_rho_up = data_func.filter_N(time_rho, Rho, avg_windows[0])
    filtered_rho_dw = data_func.filter_N(time_rho, Rho, avg_windows[1])

    density_up = calc_func.mean_std(filtered_rho_up)
    density_dw = calc_func.mean_std(filtered_rho_dw)

    #print("There are",len(rho),"data values of the bulk flow velocity in this date range.")
    #print("There are",len(filtered_rho_up),"data values of the sliced bulk flow velocity in this time range.\n")

    data_func.print_data("Upstream Density", density_up, rho_unit)
    data_func.print_data("Downstream Density", density_dw, rho_unit)
    print("")
    
    return time_rho, rho, rho_unit, density_up, density_dw


def one_window_temp(name_spacecraft, start_date, end_date, avg_windows, rho):

    #--------------- Retrieves temperature data ---------------#
    time_pres, temp, temp_unit = data_func.get_temperature(name_spacecraft, start_date, end_date)

    Temp = np.array([Temp[0] for Temp in temp])
    Rho = np.array([Density[0] for Density in rho])

    if name_spacecraft=="WIND":
        Pres = calc_func.pressure_vth(Rho*con.m_p, Temp)
    else:
        Pres = calc_func.pressure_temp(Rho, Temp)

    # Defining the averaging windows for p_up, p_dw
    filtered_pres_up = data_func.filter_P(time_pres, Pres, avg_windows[0])
    filtered_pres_dw = data_func.filter_P(time_pres, Pres, avg_windows[1])

    pres_up = calc_func.mean_std(filtered_pres_up)
    pres_dw = calc_func.mean_std(filtered_pres_dw)

    #print("There are",len(Temp),"data values of the magnetic field in this date range.")
    #print("There are",len(filtered_pres_up),"data values of the sliced magnetic field in this time range.\n")

    data_func.print_data("Upstream Pressure", pres_up*10**9, "nPa")
    data_func.print_data("Downstream Pressure", pres_dw*10**9, "nPa")
    print("")
    
    return time_pres, temp, temp_unit, pres_up, pres_dw

def velocity_shock(vel_up, vel_down, dens_up, dens_down):
    """
    Function to estimate the speed of the shock.
        v_sh = Delta(rho . V)/Delta rho . n
    The Delta represents the difference in the downstream and upstream quantities.
    V is the bulk flow speed, rho is the (mass) density
    
    Input
        vel_up - the bulk flow speed in the upstream
        vel_down - the bulk flow speed in the downstream
        dens_up - the mass density in the upstream
        dens_down - the mass density in the downstream
        normal - the normal vector of the shock
    Output
        v_sh - the shock velocity
    """
    flow_down = dens_down * vel_down
    flow_up = dens_up * vel_up
    return np.subtract(flow_down, flow_up)/np.subtract(dens_down, dens_up)

def speed_shock(vec_V_up, vec_V_dw, density_up, density_dw, normal):
    """
    Function to calculate the velocity of the shock given the density and velocity upstream and downstream of the shock
    
    Input
        vec_V_up - the bulk flow speed in the upstream
        vec_V_dw - the bulk flow speed in the downstream
        density_up - the mass density in the upstream
        density_dw - the mass density in the downstream
        normal - the normal vector to be used in the calculation of the speed of the shock
    Output
        speed - the shock speed as calculated using the 'velocity_shock' method
    """
    shock_vel = velocity_shock(vec_V_up, vec_V_dw, density_up, density_dw)
    return abs(np.dot(shock_vel, normal))

def one_window_shock_speed(vec_B_up, vec_B_dw, vec_V_up, vec_V_dw, density_up, density_dw, method_names, V_unit="km/s"):

    #--------------- Computing the normals: MC, VC, MX1, MX2, MX3 ---------------#

    # Define the delta vectors used in all methods
    dB = (vec_B_dw) - (vec_B_up)
    dV = (vec_V_dw) - (vec_V_up)
    
    function_names = [calc_func.normal_MC, calc_func.normal_MX1, calc_func.normal_MX2, calc_func.normal_MX3, calc_func.normal_VC] # Arguments for the normal_method() function
    normals = []
    angles = []
    speeds = []

    # Calculate and print normal vectors and angles
    # Calculate and print the velocity of the shocks
    for i in range(len(method_names)):
        normal_i, angle_i = calc_func.normal_method(vec_B_up, vec_B_dw, dB, dV, function_names[i])
        normals.append(normal_i) ; angles.append(angle_i)
        data_func.print_data("Normal", normal_i, method=method_names[i])
        print("Angle: {:fP} degrees".format(angle_i))
        
        speed = speed_shock(vec_V_up, vec_V_dw, density_up, density_dw, normals[i])
        speeds.append(speed)
        data_func.print_data("Speed", speed, V_unit)
        print("") 
        
def SP_one_window(vec_B_up, vec_B_dw, vec_V_up, vec_V_dw):
    #--------------- Computing the normals using SerPyShock for one window: MC, MX1, MX2, MX3 ---------------#

    nom_vec_B_up = data_func.nom_vals(vec_B_up)
    nom_vec_B_dw = data_func.nom_vals(vec_B_dw)
    nom_vec_V_up = data_func.nom_vals(vec_V_up)
    nom_vec_V_dw = data_func.nom_vals(vec_V_dw)

    print("Using SP:\n")
    # MC Method
    MC = SP.calc_MC(nom_vec_B_up, nom_vec_B_dw, "GSE")
    print("Method MC\nNormal:", MC[0], "\nAngle:", MC[1])

    # MX1 Method
    MX1 = SP.calc_MX1(nom_vec_B_up, nom_vec_B_dw, nom_vec_V_up, nom_vec_V_dw, "GSE")
    print("Method MX1\nNormal:", MX1[0], "\nAngle:", MX1[1])

    # MX2 Method
    MX2 = SP.calc_MX2(nom_vec_B_up, nom_vec_B_dw, nom_vec_V_up, nom_vec_V_dw, "GSE")
    print("Method MX2\nNormal:", MX2[0], "\nAngle:", MX2[1])

    # MX3 Method
    MX3 = SP.calc_MX3(nom_vec_B_up, nom_vec_B_dw, nom_vec_V_up, nom_vec_V_dw, "GSE")
    print("Method MX3\nNormal:", MX3[0], "\nAngle:", MX3[1])
    
def extend_data(B, V, rho):

    # Determining the extended velocity vector to be used in the Trotta method calculations
    data = V

    # Number of indexes to extend
    num_indexes = np.shape(B)[0]

    # Generate an array of indices
    indices = np.arange(data.shape[0])

    # Interpolate each column separately
    interpolated_data_V = np.empty((num_indexes, data.shape[1]))
    for col in range(data.shape[1]):
        interpolator = interp1d(indices, data[:, col], kind='cubic')
        interpolated_data_V[:, col] = interpolator(np.linspace(0, indices[-1], num_indexes))

    # Make an array with the data
    indices_rho = np.linspace(0, len(rho) - 1, num=len(rho))
    new_indices_rho = np.linspace(0, len(rho) - 1, num=np.shape(B)[0])

    # Flattens the arrays
    original_data_flat = np.ravel(rho)
    indices_flat = np.ravel(indices_rho)

    # Perfors linear interpolation
    extended_rho = np.interp(new_indices_rho, indices_flat, original_data_flat)

    # Reshape extended_density
    extended_rho_reshaped = extended_rho.reshape((np.shape(B)[0], 1))
    
    return interpolated_data_V, extended_rho_reshaped

# Calculate the velocity with averaging windows
def SP_multiple_windows(times, B, V, rho, time_shock, up_shk_off, dw_shk_off, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad, method_names, V_unit="km/s"):

    #---------- Calculating with multiple averaging windows ----------#
    
    up_shk = time_shock - up_shk_off
    dw_shk = time_shock + dw_shk_off
    
    # Calculate the normal with averaging windows
    n, theta_Bn, rB, ex = SP.MX_stats(times, B, times, V, time_shock, up_shk, dw_shk, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad, "GSE")
    
    sp_normals = np.array([n.MC, n.MX1, n.MX2, n.MX3])
    sp_angles = np.array([theta_Bn.MC, theta_Bn.MX1, theta_Bn.MX2, theta_Bn.MX3])

    # Calculate and compare the average of the normal from Trotta
    avg_normals = []
    avg_angles = []
    avg_speeds = []

    print("\nTrotta - Multiple Averaging Windows:")
    for i in range(len(sp_normals)):
        print("\nMethod "+method_names[i])

        normal = sp_normals[i]
        avg_normal = calc_func.mean_std(normal, 0)
        avg_normals.append(avg_normal)
        data_func.print_data("Normal", avg_normal)

        angle = sp_angles[i]
        avg_angle = calc_func.mean_std(angle)
        avg_angles.append(avg_angle)
        data_func.print_data("Angle", avg_angle)

        window_speeds, ex = SP.Vsh_stats(data_func.nom_vals(avg_normal), times, V, rho, time_shock, up_shk, dw_shk, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad)
        avg_speed = calc_func.mean_std(window_speeds)
        avg_speeds.append(avg_speed)
        data_func.print_data("Shock Speed", avg_speed, V_unit)

    return avg_normals, avg_speeds

# Calculate the velocity with averaging windows
def SP_multiple_windows_method(times, B, V, rho, time_shock, up_shk, dw_shk, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad, method="MX3", V_unit="km/s"):

    #---------- Calculating with multiple averaging windows ----------#
    
    # Calculate the normal with averaging windows
    n, theta_Bn, rB, ex = SP.MX_stats(times, B, times, V, time_shock, up_shk, dw_shk, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad, "GSE")
    
        # Creating a dictionary to store the options of satellites and the data paths 
    vars_dict = {
        "MC": (n.MC, theta_Bn.MC),
        "MX1": (n.MX1, theta_Bn.MX1),
        "MX2": (n.MX2, theta_Bn.MX2),
        "MX3": (n.MX3, theta_Bn.MX3)
    }
    # Retrives the path based on the spacecraft input
    params = vars_dict.get(method)
    if params is not None:
        (sp_normals, sp_angles) = params
    else:
        raise Exception("Method not found")
    
    print("\nTrotta Multiple Averaging Windows:\n")
    
    avg_normal = calc_func.mean_std(sp_normals, 0)
    avg_normal_unit = avg_normal / calc_func.magnitude(avg_normal)
    data_func.print_data("Normal", avg_normal_unit)

    avg_angle = calc_func.mean_std(sp_angles)
    data_func.print_data("Angle", avg_angle)

    window_speeds, ex = SP.Vsh_stats(data_func.nom_vals(avg_normal), times, V, rho, time_shock, up_shk, dw_shk, min_up_dur, max_up_dur, min_dw_dur, max_dw_dur, tcad)
    avg_speed = calc_func.mean_std(window_speeds)
    data_func.print_data("Shock Speed", avg_speed, V_unit)

    return avg_normal_unit, avg_speed