# Random Magnetosheath point propegation
# Last updated: 10/07/2023
# Author: Christopher Butcher
# Orginal Date: 10/07/2023
#
# NECCISARY FILES TO RUN:
# KNN_swi_Vall_k50000.pkl
# And the space folder from (https://github.com/LaboratoryOfPlasmaPhysics/space)
#
#
# ========================================================== Imports ========================================================
#
# My modules:
import Coordinate_systems as co
import Shocks_propagation as shks

# - - - - - - - - - - - - - - - - - - - - - -  Useful Modules - - - - - - - - - - - - - - - - 
# To make plots inteactive use # before
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - For g-calculation - - - - - - - - - - - - - - - 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
#
#
# This will need changing for user of the module
#
import sys
sys.path.append('/Users/christopherbutcher/Downloads/Summer Masterclass/space')
import space
from space.models import planetary as smp
from space import smath as sm
from space.coordinates import coordinates as scc
from space import utils as su
#
#
#  ESSENTIAL MODULE DEF BEFORE FUNC
# magnetosheath model using shue98 et jelinek2012
msh = smp.Magnetosheath(magnetopause='mp_shue1998', bow_shock ='bs_jelinek2012')

#kNN_location = '/Users/christopherbutcher/Downloads/Summer Masterclass/KNN_swi_Vall_k50000.pkl'
interpvall = pd.read_pickle('/Users/christopherbutcher/Downloads/Summer Masterclass/KNN_swi_Vall_k50000.pkl')

# ===========================================================================================================================
# =================================== This can be removed after script has been ran once ====================================
# ===========================================================================================================================

import json

Shocks_from_Berdichevsky = {
        '1': {'Vs_up':(120,20), 'Vs': (-443,-76,25), 'phi_shock': (214,3), 'theta_shock': (17,3), 'phi_B': (103,5), 'theta_B': (15,11), 'mag_B': (10,1.5)},
        '2': {'Vs_up':(61,10), 'Vs': (-376,-10,9), 'phi_shock': (190,12), 'theta_shock': (10,8), 'phi_B': ('Na','Na'), 'theta_B': ('Na','Na'), 'mag_B': ('Na','Na')},
        '3': {'Vs_up':(94,7), 'Vs': (-474,-8,33), 'phi_shock': (202,2), 'theta_shock': (44,2), 'phi_B': (270,1), 'theta_B': (17,0.5), 'mag_B': (4.1,0.2)},
        '5': {'Vs_up':(90,5), 'Vs': (-375,4,-30), 'phi_shock': (170,6), 'theta_shock': (-15,4), 'phi_B': (84,2), 'theta_B': (-41,4), 'mag_B': (3.9,0.2)},
        '10': {'Vs_up':(80,8), 'Vs': (-386,-50,15), 'phi_shock': (206,14), 'theta_shock': (2,2), 'phi_B': (176,19), 'theta_B': (19,14), 'mag_B': (2.0,0.4)},
        '11': {'Vs_up':(75,10), 'Vs': (-381,12,44), 'phi_shock': (147,7), 'theta_shock': (30,2), 'phi_B': (290,12), 'theta_B': (20,14), 'mag_B': (2.1,0.5)},
        '12': {'Vs_up':(73,8), 'Vs': (-385,-6,-20), 'phi_shock': (185,5), 'theta_shock': (-21,5), 'phi_B': (250,4), 'theta_B': (20,13), 'mag_B': (6.6,1.0)},
        '13': {'Vs_up':(110,20), 'Vs': (-408,-36,-18), 'phi_shock': (202,9), 'theta_shock': (-9,7), 'phi_B': (85,68), 'theta_B': (0,90), 'mag_B': (0.7,1.2)},
        '14': {'Vs_up':(137,5), 'Vs': (-467,72,52), 'phi_shock': (128,0), 'theta_shock': (21,0), 'phi_B': ('Na','Na'), 'theta_B': ('Na','Na'), 'mag_B': ('Na','Na')},
        '15': {'Vs_up':(110,25), 'Vs': (-433,-30,84), 'phi_shock': (120,7), 'theta_shock': (48,3), 'phi_B': (104,3), 'theta_B': (0,6), 'mag_B': (4.4,0.1)},
        '16': {'Vs_up':(60,10), 'Vs': (-342,1,-84), 'phi_shock': (182,10), 'theta_shock': (-48,12), 'phi_B': (87,3), 'theta_B': (-10,3), 'mag_B': (3.6,0.2)},
        '17': {'Vs_up':(39,7), 'Vs': (-346,-24,-1), 'phi_shock': (162,5), 'theta_shock': (13,3), 'phi_B': (316,4), 'theta_B': (13,8), 'mag_B': (3.4,0.4)},
        '18': {'Vs_up':(100,10), 'Vs': (-416,-85,-54), 'phi_shock': (205,4), 'theta_shock': (-15,2), 'phi_B': (141,16), 'theta_B': (-9,14), 'mag_B': (6.5,1.6)},
# Prev in data (Reverse Shock)'19': {'Vs_up':(125,15), 'Vs': (-509,-28,71), 'phi_shock': (,), 'theta_shock': (,)},
        '20': {'Vs_up':(52,8), 'Vs': (-391,-11,7), 'phi_shock': (188,6), 'theta_shock': (16,12), 'phi_B': (17,0.6), 'theta_B': (23,2), 'mag_B': (4.1,0.2)},
        '21': {'Vs_up':(45,12), 'Vs': (-422,-3,20), 'phi_shock': (185,7), 'theta_shock': (45,15), 'phi_B': (352,6), 'theta_B': (11,3), 'mag_B': (3.0,0.2)},
        '22': {'Vs_up':(50,15), 'Vs': (-359,-9,45), 'phi_shock': (156,24), 'theta_shock': (55,7), 'phi_B': (4.7,0.1), 'theta_B': (10,3), 'mag_B': (2.7,0.3)},
        '23': {'Vs_up':(52,10), 'Vs': (-380,-20,22), 'phi_shock': (202,2), 'theta_shock': (25,8), 'phi_B': (112,2), 'theta_B': (25,2), 'mag_B': (4.4,0.2)},
        '24': {'Vs_up':(70,7), 'Vs': (-318,-49,49), 'phi_shock': (213,3), 'theta_shock': (50,3), 'phi_B': (138,5), 'theta_B': (15,2), 'mag_B': (5.7,0.4)},
        '25': {'Vs_up':(75,15), 'Vs': (-478,-5,-6), 'phi_shock': (168,5), 'theta_shock': (4,4), 'phi_B': (253,3), 'theta_B': (-15,7), 'mag_B': (7.6,0.8)},
        '26': {'Vs_up':(60,20), 'Vs': (-355,-32,-22), 'phi_shock': (187,10), 'theta_shock': (-24,5), 'phi_B': (143,10), 'theta_B': (9,5), 'mag_B': (2.0,0.2)},
        '28': {'Vs_up':(55,16), 'Vs': (-329,-11,15), 'phi_shock': (169,3), 'theta_shock': (21,2), 'phi_B': (149,3), 'theta_B': (36,10), 'mag_B': (4.7,0.6)},
        '29': {'Vs_up':(40,10), 'Vs': (-370,-20,0.5), 'phi_shock': (194,10), 'theta_shock': (-6,7), 'phi_B': (164,19), 'theta_B': (26,27), 'mag_B': (2.2,0.8)},
        '30': {'Vs_up':(75,8), 'Vs': (-439,-31,-18), 'phi_shock': (200,3), 'theta_shock': (-30,3), 'phi_B': (161,4), 'theta_B': (8,7), 'mag_B': (2.2,0.3)},
        '32': {'Vs_up':(115,13), 'Vs': (-632,44,-26), 'phi_shock': (197,26), 'theta_shock': (5,5), 'phi_B': (204,9), 'theta_B': (30,11), 'mag_B': (3.6,0.6)},
        '33': {'Vs_up':(77,6), 'Vs': (-393,-17,24), 'phi_shock': (184,1), 'theta_shock': (-10,1), 'phi_B': (348,6), 'theta_B': (29,8), 'mag_B': (4.5,0.5)},
        '34': {'Vs_up':(53,3), 'Vs': (-337,-16,43), 'phi_shock': (194,1), 'theta_shock': (52,0.5), 'phi_B': (72,2), 'theta_B': (29,8), 'mag_B': (3.1,0.4)},
        '35': {'Vs_up':(72,10), 'Vs': (-378,-14,53), 'phi_shock': (199,8), 'theta_shock': (58,3), 'phi_B': (291,17), 'theta_B': (-20,24), 'mag_B': (2.0,0.6)},
        '36': {'Vs_up':(91,10), 'Vs': (-359,-5.5,77), 'phi_shock': (160,7), 'theta_shock': (51,5), 'phi_B': (330,20), 'theta_B': (45,12), 'mag_B': (7.0,1.5)},
        '38': {'Vs_up':(64,6), 'Vs': (-372,-17,43), 'phi_shock': (217,2), 'theta_shock': (49,2), 'phi_B': (14,0.5), 'theta_B': (-11,32), 'mag_B': (2.0,0.8)},
        '39': {'Vs_up':(121,10), 'Vs': (-422,-70,-58), 'phi_shock': (208,0.5), 'theta_shock': (-19,1), 'phi_B': (113,2), 'theta_B': (-17,5), 'mag_B': (7.2,0.5)},
        '40': {'Vs_up':(58,6), 'Vs': (-344,16,-32), 'phi_shock': (190,20), 'theta_shock': (-15,12), 'phi_B': (142,9), 'theta_B': (-28,11), 'mag_B': (3.4,0.8)},
        '41': {'Vs_up':(73,7), 'Vs': (-363,-6,-32), 'phi_shock': (185,2), 'theta_shock': (-20,1), 'phi_B': (103,18), 'theta_B': (37,16), 'mag_B': (4.8,1.3)},
        '42': {'Vs_up':(60,9), 'Vs': (-339,-10,-36), 'phi_shock': (176,14), 'theta_shock': (-26,7), 'phi_B': (158,1), 'theta_B': (-8,21), 'mag_B': (2.5,0.5)}
                            }

with open('shocks_from_Berdichevsky.json', 'w') as f:
    json.dump(Shocks_from_Berdichevsky, f)

# ============================================================================================================================
# ========================== Storing values of solar wind and shock data in GSE coordinates (from JSON file) =================
# ============================================================================================================================
#
# Storing values of solar wind and shock data in GSE coordinates (from JSON file)
#
v_sw_cart_GSE_shocks   = []
u_sw_cart_GSE_shocks   = []
B_field_GSE_shocks     = []
shock_index            = [] # This allows you to give later refernece to which shock in Berdichevsky produced this


for shock in Shocks_from_Berdichevsky:
    
    
    # -------------- B-field Direction ------------
    #
    # Some do not have a calculated B-field
    if Shocks_from_Berdichevsky[shock]['mag_B'][0]=='Na':
        
        pass
        
    else:
        
        # ---------------------------------- Solar wind --------------------------
        # solar wind GSE cartisian vector
        u_sw_GSE = Shocks_from_Berdichevsky[shock]['Vs']
        
        # Storing solar wind data
        u_sw_cart_GSE_shocks.append(np.asarray(u_sw_GSE))
        
        
        # ---------------------------------- Shock data --------------------------
        # shock normal vector (in GSE spherical polars)     + solar wind
        v_sw_mag = Shocks_from_Berdichevsky[shock]['Vs_up'][0] + co.mag(u_sw_GSE)
        
        # Shock normal in GSE cartisans (conversion)
        v_sw_cart_GSE = co.spher2cart_GSE(v_sw_mag,co.deg2rad(Shocks_from_Berdichevsky[shock]['theta_shock'][0]),co.deg2rad(Shocks_from_Berdichevsky[shock]['phi_shock'][0]))
        
        # Storing v_sw directions
        v_sw_cart_GSE_shocks.append(v_sw_cart_GSE)
        
        # ----------------------------------- Magnetic field data ----------------
        
        B_field_GSE_shocks.append(co.spher2cart_GSE(Shocks_from_Berdichevsky[shock]['mag_B'][0],co.deg2rad(Shocks_from_Berdichevsky[shock]['theta_B'][0]),co.deg2rad(Shocks_from_Berdichevsky[shock]['phi_B'][0])))
        
        
        # ------------------ Shock Index -----------------------------------------
        # Each shocks index
        shock_index.append(int(shock))
        
# ==========================================================================================================================
# ===================================== Transforming GSE coordinates to SWI coordinates ====================================
# ==========================================================================================================================

u_sw_cart_SWI_shocks      = []
v_sw_cart_SWI_shocks      = []
#initial_cart_shock_points = []

for i in range(0,len(shock_index),1):
    
    #                                     x_GSE                   y_GSE                      z_GSE                          B_x                     B_y                          B_z
    SWI_base_matrix = co.swi_base(u_sw_cart_GSE_shocks[i][0],u_sw_cart_GSE_shocks[i][1],u_sw_cart_GSE_shocks[i][2],B_field_GSE_shocks[i][0],B_field_GSE_shocks[i][1],B_field_GSE_shocks[i][2])
    
    
    # ------------------------- Solar wind vectors to SWI coordinates ------------------
    u_sw_cart_SWI = co.GSE2SWI(u_sw_cart_GSE_shocks[i],SWI_base_matrix)
    
    u_sw_cart_SWI_shocks.append(u_sw_cart_SWI)
    
    # ------------------------------- Shock normal vectors -----------------------------
    v_sw_cart_SWI = co.GSE2SWI(v_sw_cart_GSE_shocks[i],SWI_base_matrix)
    
    v_sw_cart_SWI_shocks.append(v_sw_cart_SWI)


# ==========================================================================================================================
# ==================================================== Defining functions ==================================================
# ==========================================================================================================================

# ------------------------------- Function is used to calculate the g-value for each point from arrays contining coordinates
def g_values(x_array,y_array,z_array):
    
    g_vals = []
    
    for i in range(0,len(x_array),1):
        # usage for a single point:
        x,y,z = x_array[i],y_array[i],z_array[i]
        pos = np.asarray([[x,y,z]])
        vx,vy,vz = interpvall.predict(pos).T
        #vx,vy,vz
        
        g = np.asarray([vx[0],vy[0],vz[0]])
        
        #g_vals.append(g)
        
        g_vals.append(co.mag(g))
        
    return g_vals


# -------------------------------- This function calculates the additive hypotheses
def v_msh_add_hyp(u_sw,g_val,v_sw):
    
    v_MSH = v_sw + (g_val-1)*u_sw
    
    return v_MSH


# -------------------------------- Changing speeds into units of r_E/s
r_E = 6378.16 # km

# -------------------------------- This will be used as the maximum distance 
r_bs_nose = msh.bow_shock(0,0)[0]


# ==========================================================================================================================
# ==================================================== f-value calculations ================================================
# ==========================================================================================================================


# -------------------------------- the number of random samples (Choose in terminal)
print('How many points would you like to sample')
n_shock_points = input()

# Converting to integer
n_shock_points = int(n_shock_points)
#n_shock_points = 10


# -------------------------------- Choosing the step size for path integral (Choose in terminal)
print('Choose the step size (for the path integral)')
step_size = input()

# Converting to float
step_size = float(step_size)
#step_size = 0.5 # in r_E

# --------------------------------- Choosing maximum theta value
print('What maximum theta value do you want:')
theta_m = input()

# Converting to float
theta_m = float(theta_m)


# Calculating where x-ends for chosen theta_m
x_end = co.mag(msh.bow_shock(co.deg2rad(theta_m),0))*np.cos(np.pi-co.deg2rad(theta_m))
print('The Sampeling box ends at x_min = ', -x_end)

# -------------------------------- Assumed direction of prop (-x direction)
direction_of_prop = np.array([1,0,0]) # made to be positive as this array is later reversed

# Working out the x-stop value from theta max value (specified in random generator function)
theta_max = co.deg2rad(theta_m)
Y_max = co.mag(msh.bow_shock(theta_max,0))
X_min = Y_max*np.cos(theta_max)
x_stop = X_min

# Initilise counter
point_counter = 0

# storage of f-values
f_mult_vals       = []
f_add_vals        = []
shock_point_index = []

# -------------------------------- Progresss Bar
import time
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Initial call
printProgressBar(0,n_shock_points , prefix = 'Percentage of Samples:', suffix = 'Complete', length = 50)

#items = list(range(0, 57))
#l = len(items)


# ==========================================================================================================================
# ================================================== Random point generator ================================================
# ==========================================================================================================================

# 
# Random MSH Point -  x_limit corresponds to the points you want to exclude
#                     (usually x_limit=0 i.e. only allows x>0)
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



# ==========================================================================================================================
# ================================================== f-value calculator loop ===============================================
# ==========================================================================================================================
import array

# Generating random points on the bow shock:
while point_counter <= n_shock_points:

    # ------------------------ Getting a random point inside msh --------------------

    # -------- New -------------
    # Get a random point in msh
    ran_pos = rand_msh_point_SWI(theta_m)

    # Find the initial Bow shock that is colinear to this point in the x-axis
    points_of_prop = shks.propegation_V2(ran_pos,direction_of_prop,step_size,x_stop)


    # Reverse list as it originally starts at a random msh point
    points_of_prop[0].reverse()
    points_of_prop[1].reverse()
    points_of_prop[2].reverse()
    

    d_rand = points_of_prop[0][0] - ran_pos[0]
    
    # --------------------------- Calculate g-along these lines of prop -------------------
    
    g_at_points = g_values(points_of_prop[0],points_of_prop[1],points_of_prop[2])
    
    
    # --------------------------- Use g to calculate the time integral --------------------
    
    # Temporary storage for the f-values calculate from bnerdichevsky data
    f_mult_each_shock = []
    f_add_each_shock = []
    
    # Initilise times
    mult_time = 0
    add_time  = 0
    
    # ============================ Using values from each shock in Berdichevsky ==============
    for a in range(0,len(v_sw_cart_SWI_shocks),1):
        
        shock_mag      = co.mag(v_sw_cart_SWI_shocks[a])
        solar_wind_mag = co.mag(u_sw_cart_SWI_shocks[a])
        
        t2 = add_time+1
        # ============== Calculation of time to propegate along line ========
        if len(g_at_points)==0:
            continue              # this is caused by step inaccuracy and thus cannot count as a shock propegation
        
        for j in range(0,len(g_at_points),1):
            
            # Sum up dt for each propegation --- Multiply by r_E for step distance to be in km ---
            
            mult_time = mult_time + (step_size*r_E)/(g_at_points[j]*(shock_mag))
            
            add_time  = add_time + (step_size*r_E)/(v_msh_add_hyp(solar_wind_mag,g_at_points[j],shock_mag))
            
            
        # Calculate the magnetosheath velocity
        
        v_msh_mult = (d_rand*r_E)/mult_time
        
        v_msh_add  = (d_rand*r_E)/add_time
        
        # Calculate f-value
        f_mult = v_msh_mult/shock_mag
        
        f_add  = v_msh_add/shock_mag
        
        # Store these for each initial shock point
        
        f_mult_each_shock.append(f_mult)
        
        f_add_each_shock.append(f_add)
        
        # Re-Initilise times
        mult_time = 0
        add_time  = 0
        
    # Storing each berdichevsky shocks f-value for each shock point
    f_mult_vals.append(f_mult_each_shock)
    
    f_add_vals.append(f_add_each_shock) # Stores [ shock_point 1:[berdichevsky_shock 1,berdichevsky_shock 2]
    #                                              shock_point 2:[berdichevsky_shock 1,berdichevsky_shock 2] ]
    
    # Counting the number of calculated shock points
    point_counter = point_counter + 1
    shock_point_index.append(point_counter)
    
    # Update progress
    printProgressBar(point_counter,n_shock_points+1, prefix = 'Percentage of Samples:', suffix = 'Complete', length = 50)

# ==========================================================================================================================
# ============================================ f-value vs Shock propegation ================================================
# ==========================================================================================================================
#
# Plotting each value for f-for each shock point
#
# Figure things
fig = plt.figure()
ax = fig.add_subplot()
#

for i in range(0,len(f_add_vals),1):
    
    for j in range(0,len(f_add_vals[i]),1):
        
        col1 = (j/len(f_add_vals[i]),0,j/len(f_add_vals[i]))
        
        col2 = (0,j/len(f_add_vals[i]),0)
        
        ax.scatter(shock_point_index[i],f_add_vals[i][j],color=col1,marker='+')
        
        ax.scatter(shock_point_index[i],f_mult_vals[i][j],color=col2,marker='x')
        

# setting title and labels
ax.set_title("random bow shock points")
ax.set_xlabel('Random shock point')
ax.set_ylabel('f-value')
 
# displaying the plot
plt.show()

# ==========================================================================================================================
# =================================================== f-value histogram ====================================================
# ==========================================================================================================================

# Plotting each value for f-for each shock point

# Figure things
#
# creating figure
fig = plt.figure()
ax = fig.add_subplot()
#

add_f_array = []
mult_f_array = []

for i in range(0,len(f_add_vals),1):
    
    for j in range(0,len(f_add_vals[i]),1):
        
        f_add = f_add_vals[i][j]
        add_f_array.append(f_add)
        
        
        f_mult = f_mult_vals[i][j]
        mult_f_array.append(f_mult)

        
# Plotting histograms
ax.hist(add_f_array,20,label='additive')
ax.hist(mult_f_array,20,label='multiplicative',alpha=0.5)


# setting title and labels
ax.set_title("Randomly sampled points of propagation")
ax.set_xlabel('f-value ($v_{msh}/v_{sw}$)')
ax.set_ylabel('Count')

plt.legend()
 
# displaying the plot
plt.show()