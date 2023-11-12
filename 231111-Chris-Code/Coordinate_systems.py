# Coordinate_systems.py
# Version 1.3
# Last Updated: 05.08.2023
# Christopher Butcher
#
# This module contains functions that can be used to convert between coordinate systems
# for the interplanatary shock code
# ---------------------------------
# When Passing arrays into functions,
# make sure they are np.array([]) defined!!!!
# ---------------------------------
#
# Module Contents:
#      
#    - Imports
#
#    - Vectors:
#        - mag
#    
#    - Angle Conversions:
#        - deg2rad
#        - rad2deg
#      
#    - GSE Coordinates:
#        - spher2cart_GSE
#        - cart2spher_GSE
#    
#    
#    - SWI Coordinates:
#        - SWI Spherical polar transforms:
#            - spher2cart_SWI
#            - cart2spher_SWI
#
#    - SWI and GSE Transformations:
#        - GSE2SWI
#        - SWI2GSE
#
#    - Welle's coordinates Module functions:
#        - swi_base
#
#    - Disgarded functions:
#        - SWI_BASIS_MATRIX
#
# =====================================================
# ==================== IMPORTS ========================
# =====================================================
#
# These Imports are for use in the functions of the module
#
import numpy as np
#
# =====================================================
# ==================== Vectors ========================
# =====================================================
#
# Expects a vector x and returns the pythagorian norm
def mag(x):
    
    a = np.sqrt(np.dot(x,x))
    
    return a
#
#
# =====================================================
# ================= Angle Conversions =================
# =====================================================
#
# Converts Degrees to radians
def deg2rad(d):
    r = d*np.pi/180
    return r
#
# 
# Converts radians to degrees
def rad2deg(r):
    d = r*180/np.pi
    return d
#
# =====================================================
# ================== GSE Coordinates ==================
# =====================================================
# 
# GSE (Geocentric Solar Ecliptic system)
# This has its X-axis pointing from the Earth toward the Sun and its Y-axis is chosen to be in the ecliptic plane pointing towards dusk (thus opposing planetary motion). Its Z-axis is parallel to the ecliptic pole. Relative to an inertial system this system has a yearly rotation.
#
# theta = latitiude angle
# phi   = longitudinal angle
# 
# theta is defined as the elevation angle between the x-axis and the position r
# phi is defined as the angle between the x-axis and its xy-projection in the xy plane.
#
def spher2cart_GSE(r,theta,phi):
    
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    
    return np.array([x,y,z])
#
#
def cart2spher_GSE(x,y,z):
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    theta = np.arccos(z/r)
    
    phi = np.arctan2(y,x) # Note use of arctan2 function (see https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html)
    
    return np.array([r,theta,phi])
#
# =====================================================
# ================== SWI Coordinates ==================
# =====================================================
#
# SWI (Solar Wind Interplanetary magnetic field coordinate system)
#
# The x-axis points directly towards the incoming [[Solar wind]] also known as being co-linear ( SW travelling in the -x direction). The y-axis is determined from the $\mathbf{B}$ field of the solar wind, and the z-axis is then perpendicular to this (using Gram-Shmidt or cross product).
#
#
# 
#
def spher2cart_SWI(r,theta,phi):
    
    # components calculation
    x = r*np.cos(theta)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.sin(theta)*np.cos(phi)
    
    return np.array([x,y,z])
#
#
def cart2spher_SWI(x,y,z):
    
    r = np.sqrt(x**2 + y**2 +z**2)
    
    theta = np.arccos(x/r)
    
    phi = np.arctan2(y,z) # arctan2 prevents infinities
    
    return np.array([r,theta,phi])
#
#
# =====================================================
# =========== SWI and GSE Transformations =============
# =====================================================
#
#
# Coordinate conversion for (GSE -> SWI)
# (This is not a essential function however it makes the code more clear as to what vector is being transformed)
def GSE2SWI(GSE_vector,SWI_BASIS):
    
    # SWI_BASIS is expected to be a matrix [[x_SWI],[y_SWI],[z_SWI]]
    # GSE is assumed to be the identity operator so we dont have to multiply matricies
    # I = [[1,0,0],[0,1,0],[0,0,1]] = GSE_BASIS
    
    SWI_vector = np.dot(SWI_BASIS,GSE_vector)
    
    return np.array(SWI_vector)
#
#
# Coordinate conversion for (SWI -> GSE)
# Note this would be more efficent to define and use in code, this function can be used
# as a outline if required to speed up calculation
def SWI2GSE(SWI_vector,SWI_BASIS):
    
    # SWI_BASIS is expected to be a matrix [[x_SWI],[y_SWI],[z_SWI]]
    # GSE is the identity operator so we dont have to multiply matricies
    # I = [[1,0,0],[0,1,0],[0,0,1]] = GSE_BASIS
    
    # Inverse of SWI matrix
    SWI_Inverse = np.linalg.inv(SWI_BASIS)
    
    GSE_vector = np.dot(SWI_Inverse,SWI_vector)
    
    return np.array(GSE_vector)
#
#
# ===============================================================================
# ====================== Welle's coordinates Module functions ===================
# ===============================================================================
#
# This function will output the SWI Basis matrix that can be used to transform between
# GSE coordinates systems
#
def swi_base(vx, vy, vz, bx, by, bz):
    coa_zhang19 = np.arctan(bx / np.sqrt(by ** 2 + bz ** 2)) # This is stated differnetly in paper? (Page 3)
    sign = np.sign(coa_zhang19)
    if isinstance(coa_zhang19, float) or isinstance(coa_zhang19, int):
        if coa_zhang19 == 0:
            sign = np.sign(by)
            if by == 0:
                sign = np.sign(bz)
    else:
        sign[coa_zhang19 == 0] = np.sign(by[coa_zhang19 == 0])
        sign[(coa_zhang19 == 0) & (by == 0)] = np.sign(bz[(coa_zhang19 == 0) & (by == 0)])

    V = mag([vx, vy, vz])
    X = np.array([-vx / V, -vy / V, -vz / V])
    B = np.array([sign * bx, sign * by, sign * bz])
    Z = np.cross(X.T, B.T).T
    Z_norm = mag(Z)
    Z = np.array([Z[0] / Z_norm, Z[1] / Z_norm, Z[2] / Z_norm])
    X, Z = X.T, Z.T
    Y = np.cross(Z, X)
    return np.array([X, Y, Z])
#
#
# ===============================================================================
# ================================= Disgarded functions =========================
# ===============================================================================
#
# Reason for Disgard:
# Uses Gram Shmidt as opposed to Convention of using the B-field
#
# This Function will find a matrix containing the SWI basis vectors
# in terms of the GSE Cartisian basis.
# This matrix is also the transformation matrix used to convert from (GSE->SWI)
#
# SWI BASIS coordinates in GSE
# def SWI_BASIS_MATRIX(solar_wind,B_field):
    
#     # GSE Assumption
#     x_GSE = np.array([1,0,0])
#     y_GSE = np.array([0,1,0])
#     z_GSE = np.array([0,0,1])
    
#     # x def
#     x_SWI = -(solar_wind)/mag(solar_wind)
    
#     # The y direction is chosen according to the B_field of the solar wind
#     # 
    
#     # ---------------------- Wrong way with Gram Shmidt --------------------
#     #     # y def
#     #     y_dir_SWI = y_GSE - (np.dot(y_GSE,x_SWI))*x_SWI
#     #     y_SWI = y_dir_SWI/(mag(y_dir_SWI))
    
#     #     # z def
#     #     z_dir_SWI = z_GSE - (np.dot(z_GSE,x_SWI)*x_SWI)-(np.dot(z_GSE,y_SWI)*y_SWI)
#     #     z_SWI = z_dir_SWI/mag(z_dir_SWI)
#     # -----------------------------------------------------------------------
    
#     return np.array([x_SWI,y_SWI,z_SWI])
#
#