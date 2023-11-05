""" Importing key libraries and modules """
import numpy as np
from uncertainties import ufloat # for managing errors in the data
from uncertainties import unumpy

import scipy.constants as con
from data_functions import *
     
# Function to calculate the conserved Rankine-Hugoniot quantities using the average upstream and downstream parameters   
def rankine_1(B_up, B_dw, n):
    """
    Function to calculate the average upstream and downstream quantities for RH equation 1.
    In this function, _n refers to normal to the shock front.
    
    RH 1 -- B_n(up) = B_n(down).
        where B_n = B . n
    
    Input
        B_up - the average upstream magnetic field density
        B_dw - the average downstream magnetic field density
        n - the normal vector
    Output
        B_nu, B_nd - the component of the upstream, downstream magnetic field density along the direction of the normal vector
    """
    up_field = B_up @ n # this represents the dot product, i.e. B_n = B . n
    down_field = B_dw @ n
    
    return up_field, down_field

# Function to calculate the conserved Rankine-Hugoniot quantities using the average upstream and downstream parameters 
def rankine_2(rho_up, rho_dw, V_up, V_dw, n, v_sh=0):
    """
    Function to calculate the average upstream and downstream quantities for RH equation 2.
    In this function, _n refers to normal to the shock front.
    
    RH 2 -- rho V_n (up) = rho V_n (down).
        where V_n = V . n
    However, we need to apply the RH equation in the frame of the shock.
    To do this, we subtract the shock velocity from all velocity parameters.
    
    Input
        rho_up - the average upstream number density
        rho_dw - the average downstream number density
        V_up - the average upstream bulk flow velocity
        V_up - the average upstream bulk flow velocity
        n - the normal vector
        v_sh - the speed of the shock, (in the direction of n by definition)
    Output
        flow_u, flow_d - the mass flow upstream and downstream of the shock
    """
    # The bulk flow speeds transformed to be in the frame of the shock
    # By definition, the shock speed is in the direction of the normal
    # So we subtract the shock velocity from the bulk flow velocity vectors
    
    # Transform velocities to frame of the shock
    V_n_up = V_up @ n - v_sh
    V_n_dw = V_dw @ n - v_sh
    
    up_flow = rho_up*V_n_up
    down_flow = rho_dw*V_n_dw
    
    return up_flow, down_flow    

# Function to calculate the conserved Rankine-Hugoniot quantities using the average upstream and downstream parameters 
def rankine_3(B_up, B_dw, V_up, V_dw, n, v_sh=0):
    """
    Function to calculate the average upstream and downstream quantities for RH equation 3.
    In this function, _n refers to normal to the shock front, _t refers to transverse to the shock front.
    
    RH 3 -- B_n V_t - V_n B_t (up) = B_n V_t - V_n B_t (down).
        where V_n = V . n
              B_n = B . n
              V_t = V - V_n
              B_t = B - B_n
    However, we need to apply the RH equation in the frame of the shock.
    To do this, we subtract the shock velocity from all velocity parameters.
    
    Input
        B_up - the average upstream magnetic field density
        B_dw - the average downstream magnetic field density
        V_up - the average upstream bulk flow velocity
        V_up - the average upstream bulk flow velocity
        n - the normal vector
        v_sh - the speed of the shock, (in the direction of n by definition)
    Output
        field_u, field_d - the tangential component of the upstream and downstream electric field
    """
    # The bulk flow speeds transformed to be in the frame of the shock
    # By definition, the shock speed is in the direction of the normal
    # So we subtract the shock velocity from the bulk flow velocity vectors
    
    # Transform velocities to frame of the shock
    V_up = V_up - v_sh * n
    V_dw = V_dw - v_sh * n
    
    # Scalars
    B_n_up = B_up @ n
    B_n_dw = B_dw @ n
    
    V_n_up = V_up @ n
    V_n_dw = V_dw @ n
    
    # Vectors
    B_t_up = B_up - (B_up @ n) * n
    B_t_dw = B_dw - (B_dw @ n) * n
    
    V_t_up = V_up - (V_up @ n) * n
    V_t_dw = V_dw - (V_dw @ n) * n
    
    up_field = B_n_up*V_t_up - V_n_up*B_t_up
    down_field = B_n_dw*V_t_dw - V_n_dw*B_t_dw
    
    return up_field, down_field    

# Function to calculate the conserved Rankine-Hugoniot quantities using the average upstream and downstream parameters 
def rankine_4a(rho_up, V_up, B_up, p_up, rho_dw, V_dw, B_dw, p_dw, n, v_sh=0):
    """
    Function to calculate the average upstream and downstream quantities for RH equation 4a.
    In this function, _n refers to normal to the shock front.
    
    RH 4a -- rho V_n^2 + P + (B^2/2 - B_n^2)/mu0 (up) = rho V_n^2 + P + (B^2/2 - B_n^2)/mu0 (down).
        where V_n = V . n
              B_n = B . n
    However, we need to apply the RH equation in the frame of the shock.
    To do this, we subtract the shock velocity from all velocity parameters.
    
    Input
        rho_up - the average upstream number density
        rho_dw - the average downstream number density
        B_up - the average upstream magnetic field density
        B_dw - the average downstream magnetic field density
        V_up - the average upstream bulk flow velocity
        V_up - the average upstream bulk flow velocity
        p_up - the average upstream thermal pressure
        p_dw - the average downstream thermal pressure
        n - the normal vector
        v_sh - the speed of the shock, (in the direction of n by definition)
    Output
        pressure_u, pressure_d - the pressure upstream and downstream of the shock
    """
    # The bulk flow speeds transformed to be in the frame of the shock
    # By definition, the shock speed is in the direction of the normal
    # So we subtract the shock velocity from the bulk flow velocity vectors
    
    # Transform velocities to frame of the shock
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

# Function to calculate the conserved Rankine-Hugoniot quantities using the average upstream and downstream parameters 
def rankine_4b(rho_up, rho_dw, V_up, V_dw, B_up, B_dw, n, v_sh=0):
    """
    Function to calculate the average upstream and downstream quantities for RH equation 4.
    In this function, _n refers to normal to the shock front, _t refers to transverse to the shock front.
    
    RH 4b -- rho V_n V_t - (B_n B_t)/mu0 (up) = rho V_n V_t - (B_n B_t)/mu0 (down).
        where V_n = V . n
              B_n = B . n
              V_t = V - V_n
              B_t = B - B_n
    However, we need to apply the RH equation in the frame of the shock.
    To do this, we subtract the shock velocity from all velocity parameters.
    
    Input
        rho_up - the average upstream number density
        rho_dw - the average downstream number density
        B_up - the average upstream magnetic field density
        B_dw - the average downstream magnetic field density
        V_up - the average upstream bulk flow velocity
        V_up - the average upstream bulk flow velocity
        n - the normal vector
        v_sh - the speed of the shock, (in the direction of n by definition)
    Output
        pressure_u, pressure_d - the pressure upstream and downstream of the shock
    """
    # The bulk flow speeds transformed to be in the frame of the shock
    # By definition, the shock speed is in the direction of the normal
    # So we subtract the shock velocity from the bulk flow velocity vectors
    
    # Transform velocities to frame of the shock
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

# Function to 
def rankine_analysis(B, V, rho, p, n, v_sh=0):
    """
    A procedure that carries out the Rankine-Hugoniot analysis for our shock data.
    In this function, _n refers to normal to the shock front, _t refers to transverse to the shock front.
    This function calculates the continuous Rankine-Hugoniot quantities.
    
    Input
        B, V, rho, p - our shock perameters
        n - the normal vector of the shock
        v_sh - the shock speed calculated from the normal vector
    Output
        (RH1, RH2, RH3, RH4a, RH4b) - the five continuous Rankine-Hugoniot quantities for plotting.
    """
    B_n = B @ n # scalars
    B_t = np.subtract(B,np.array([b_n * n for b_n in B_n])) # vectors
    B2 = np.array([np.dot(B_i,B_i) for B_i in B])

    # The bulk flow speeds transformed to be in the frame of the shock
    # By definition, the shock speed is in the direction of the normal
    # So we subtract the shock velocity from the bulk flow velocity vectors

    # Transform velocities to frame of the shock
    V = V - v_sh * n
    V_n = V @ n # scalars
    V_t = np.subtract(V,np.array([v_n * n for v_n in V_n])) # vectors
    
    # The five RH quantities
    RH_1 = B_n
    RH_2 = np.multiply(rho,V_n)
    
    RH_3 = []
    for i in range(3): # the three vector components, x-y-z
        RH_3.append(np.subtract(np.multiply(B_n,V_t[:,i]),np.multiply(V_n,B_t[:,i])))
    RH_3 = np.array(RH_3)

    RH_4a = 10**9*((con.m_p*np.multiply(rho,V_n**2))*10**(12)+p+(np.multiply(B2/2,B_n**2)*10**(-18)/con.mu_0))

    RH_4b = []
    for i in range(3): # the three vector components, x-y-z
        RH_4b.append(10**9*(con.m_p*np.multiply(rho*V_n,V_t[:,i])*10**(12)-(np.multiply(B_n,B_t[:,i]))*10**(-18)/con.mu_0))
    RH_4b = np.array(RH_4b)

    return (RH_1, RH_2, RH_3, RH_4a, RH_4b)
    

def RH_result(upstream, downstream, unit):
    """
    Procedure to print the upstream and downstream RH quantities.
    Also calculates the difference between them to determine whether the RH equation is satisfied.
    
    Input
        upstream - the average upstream RH quantity
        downstream - the average downstream RH quantity
        unit - the unit of the RH quantity, for printing purposes
    """    
    if within_error(upstream, downstream):
        print("\nQuantity is conserved.")
    else:
        print("\nQuantity is not conserved.")
        
    difference = abs(downstream - upstream)
    print_data("Upstream", upstream, unit)
    print_data("Downstream", downstream, unit)
    print_data("Difference", difference, unit)
    
    if not isinstance(upstream, np.ndarray):
        num_sigma = difference.n/difference.s # Works out the number of standard deviations the difference is from 0
        print(f"{num_sigma:.5}", "sigma")
        
def within_error(m1, m2):
    """
    Function to return whether two measurements, m1 and m2, agree within error.
    This function works by working out the difference of the two measurements, dm = m1 - m2.
        If the error on this difference is greater than the difference,
            i.e. the difference agrees within error with 0, then that means that the two values are close enough to say that they don't disagree within error,
            i.e. we can't say they are inconsistent.
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