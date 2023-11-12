# Shocks_propagation.py
# Version 1.3
# Last Updated: 05.08.2023
# Christopher Butcher
#
# This Module contains functions used to calculate properties of a shock propegating through 
# the earths magnetosheath
#
# ----------------------------------------------------
# When Passing arrays into functions,
# make sure they are np.array([]) defined!!!!
# ----------------------------------------------------
#
#
# Module Contents:
#
#    - Imports
#
#    - Magnetosphere Module
#
#    - Hypotheses
#        - vec_add_hyp
#    
#    - Random Point inside Magnetosheath
#        - rand_msh_point_SWI
#
#    - Propegation functions:
#        - propegation_V1
#        - propegation_V2
#
#
# =====================================================
# ==================== IMPORTS ========================
# =====================================================
#
# This Module uses some functions from another module, thus they must be in the same
# directory. My prev defined modules:
import Coordinate_systems as co
#
#
import numpy as np
#
# -----------------------------------------------------
# These Imports are used specifically for Welle's Code:
#
# - - - - - - - - - For g-calculation - - - - - - - - -
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
#
# - - - - For r_magnetosheath and r_magnetopause - - - 
#
# This will need changing for user of the module
# ------------ PASTE THESE IN MAIN CODE ---------------
import sys
sys.path.append('/Users/christopherbutcher/Downloads/Summer Masterclass/space')
import space
from space.models import planetary as smp
from space import smath as sm
from space.coordinates import coordinates as scc
from space import utils as su
#
#
# =====================================================
# =============== Magnetosphere Module ================
# =====================================================
#
#  ESSENTIAL MODULE DEF BEFORE FUNC
# magnetosheath model using shue98 et jelinek2012
msh = smp.Magnetosheath(magnetopause='mp_shue1998', bow_shock ='bs_jelinek2012')
#
#
# =====================================================
# =============== Hypotheses ==========================
# =====================================================
#
#
# This will calculate the shock velocity vector for a given point inside the Magnetosheath
# And given the solarwind velocity vector (in SWI) and the shock velocity vector (in the solar wind)
#
def vec_add_hyp(v_sw,u_sw,position_cart):
    
    # usage for a single point:
    x,y,z = position_cart[0],position_cart[1],position_cart[2]
    pos = np.asarray([[x,y,z]])
    vx,vy,vz = interpvall.predict(pos).T
    #vx,vy,vz
        
    g_vec = np.asarray([vx[0],vy[0],vz[0]])
    
    v_msh = v_sw - u_sw + co.mag(u_sw)*g_vec
    
    return v_msh
#
#
# =====================================================
# =============== Random point generator ==============
# =====================================================
# 
# Random MSH Point -  max_theta is the x-limit at which you want to cut off the random points generated
def rand_msh_point_SWI(max_theta):
    
    #
    theta_max = co.deg2rad(max_theta)
    #
    # Need a value for theta from x_limit
    #
    # theta_lim = np.arccos(x_limit/) - - 
    Y_max = co.mag(msh.bow_shock(theta_max,0))*np.sin(np.pi-theta_max)
    Y_min = -Y_max
    #
    Z_max = Y_max
    Z_min = Y_min
    # Sample inside a box around x-points
    X_max = msh.bow_shock(0,0)[0]
    X_min = -co.mag(msh.bow_shock(theta_max,0))*np.cos(np.pi-theta_max)
    
    # initilise Boolian : (False means not inside msh, True means outside msh)
    inside_msh = False
    
    while inside_msh == False:
        
        # Generating random coordinates
        x_rand = np.random.uniform(X_min,X_max)
        y_rand = np.random.uniform(Y_min,Y_max)
        z_rand = np.random.uniform(Z_min,Z_max)
        
        # Changing to Spherical polars (SWI)
        spher_pos = co.cart2spher_SWI(x_rand,y_rand,z_rand)
        
        # Bow shock and magnetopause radii at same (theta,phi)
        bow_shock_dist = co.mag(msh.bow_shock(spher_pos[1],spher_pos[2]))
        mag_pause_dist = co.mag(msh.magnetopause(spher_pos[1],spher_pos[2]))
        
        cart_point = np.array([x_rand,y_rand,z_rand])
        r_rand = co.mag(cart_point)
        
        # If runs then inside msh
        if (r_rand < bow_shock_dist) and (r_rand > mag_pause_dist):
            
            inside_msh = True
            
    return cart_point

# =====================================================
# ================ Propegation functions ==============
# =====================================================
#
# ----------------- Propegation V1 --------------------
# 
# This function will move from the initial_position (NOTE: In Cartisians) to a distance away (S_dist)
# for a finite distance of ds_step
# Input: (Initial position array/list [x,y,z]) + (direction of propegation) + (S=disance of propegation,ds=step distance)
def propegation_V1(initial_position,sw_vec,S_dist,ds_step):
    
    # Finding the unit vector of u_sw
    sw_dir = sw_vec/(co.mag(sw_vec))
    
    cart_points = initial_position
    
    # Calculating number of steps
    n_steps = int(S_dist/ds_step)
    
    # Define array to store propegation points
    x_prop = np.array([])
    y_prop = np.array([])
    z_prop = np.array([])
    
    for j in range(0,n_steps,1):
        
        # Includes initial position
        x_prop.append(cart_points[0])
        y_prop.append(cart_points[1])
        z_prop.append(cart_points[2])
        
        # Moving in direction of sw_dir in steps of ds
        cart_points = cart_points + sw_dir*ds_step
    
    return np.array([x_prop,y_prop,z_prop])
#
#
# ----------------- Propegation V2 --------------------
#
#
# Version 2 will exit the propegation loop if the shock line hits:
#     - magnetopause boundary
#     - bowshock boundary
#     - Specified end x-point (In SWI)
#
# Requires importation of:
#     - Coordinate_systems.py (local)
#     - msh
#
# This propegation function takes in the initial position of the bow shock intersect
# Then propegates in the direction of the planar shock wave
#
def propegation_V2(initial_pos,sw_vec,ds_step,else_stop):
    
    # Finding the unit vector of u_sw
    sw_dir = sw_vec/(co.mag(sw_vec))
    
    cart_points = initial_pos
    
    # Define array to store propegation points
    x_prop = []
    y_prop = []
    z_prop = []
    
    # Initial Points - prevents empty defs ------------------
    # Convert to spherical coordinates
    spher_points = co.cart2spher_SWI(cart_points[0],cart_points[1],cart_points[2])
        
    # Calculate magnetopause r value at theta,phi value
    mp_rad = co.mag(msh.magnetopause(spher_points[1],spher_points[2]))
    
    # Calculate bow shock radius at theta phi value
    bs_rad = co.mag(msh.bow_shock(spher_points[1],spher_points[2]))
    
    # Magnitude of position (r value)
    r_point = co.mag(cart_points)
    # -------------------------------------------------------
    
    # LOOP TO MOVE IN SW DIR
    while (r_point > mp_rad) and (cart_points[0] > else_stop) and (r_point <= bs_rad+0.01):
    #     ^ Not in Bow MagPause + ^ Chosen x-limit         +    ^ Not outside Bow shock (+0.01 to allow initial point inclusion)
        
        # Includes initial points
        x_prop.append(cart_points[0])
        y_prop.append(cart_points[1])
        z_prop.append(cart_points[2])
        
        # Moving in direction of sw_dir in steps of ds_step
        cart_points = cart_points + sw_dir*ds_step
        
        # Convert to spherical coordinates
        spher_points = co.cart2spher_SWI(cart_points[0],cart_points[1],cart_points[2])
        
        # Calculate magnetopause r value at theta,phi value
        mp_rad = co.mag(msh.magnetopause(spher_points[1],spher_points[2]))
        
        # Calculate bow shock radius at theta phi value
        bs_rad = co.mag(msh.bow_shock(spher_points[1],spher_points[2]))
        
        # Magnitude of position (r value)
        r_point = co.mag(cart_points)
    
    if (r_point >= bs_rad):
        x_prop = []
        y_prop = []
        z_prop = []
    
    return [x_prop, y_prop, z_prop]
#
#