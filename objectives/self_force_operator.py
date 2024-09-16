import numpy as np
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve, CurrentPotential
from operator_helper import (
    norm_helper, Kdash_helper, 
    unitnormaldash, grad_helper
)
from f_b_and_k_operators import AK_helper

import sys
sys.path.insert(0, '../build')
import biest_call
sys.path.insert(0, '..')
from utils import avg_order_of_magnitude, project_arr_cylindrical

# Calculates the integrands in Robin, Volpe from a number of arrays.
# The arrays needs trimming compared to the outputs
# with a standard cp.
# The inputs are array properties of a surface object
# containing only one field period so that the code is easy to port 
# into c++.
def self_force_integrands_xyz(
    normal,
    unitnormal,
    gammadash1,
    gammadash2,
    gammadash1dash1,
    gammadash1dash2,
    gammadash2dash2,
    nfp, cp_m, cp_n,
    net_poloidal_current_amperes,
    net_toroidal_current_amperes,
    quadpoints_phi,
    quadpoints_theta,
    stellsym,
    AK_x_trimmed, bK_x_trimmed,
    current_scale):
    ''' 
    Calculates the nominators of the sheet current self-force in Robin, Volpe 2022.
    The K_y dependence is lifted outside the integrals. Therefore, the nominator 
    this function calculates are operators that acts on the QUADCOIL vector
    (scaled Phi, 1). The operator produces a 
    (n_phi_x, n_theta_x, 3(xyz, to act on Ky), 3(xyz), n_dof+1)
    After the integral, this will become a (n_phi_y, n_theta_y, 3, 3, n_dof+1)
    tensor that acts on K(y) to produce a vector with shape (n_phi_y, n_theta_y, 3, n_dof+1, n_dof+1)
    Shape: (n_phi_x, n_theta_x, 3(xyz), 3(xyz), n_dof+1(x)).

    Reminder: Do not use this with BIEST, because the x, y, z components of the vector field 
    has only one period, however many field periods that vector field has.
    ''' 
    # Add a subscript to remind that this is the unit normal
    # as a function of x in the formula.
    unitnormal_x = unitnormal
    unitnormaldash1_x, unitnormaldash2_x = unitnormaldash(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
    )
    # An assortment of useful quantities related to K
    (
        Kdash1_sv_op_x, 
        Kdash2_sv_op_x, 
        Kdash1_const_x,
        Kdash2_const_x
    ) = Kdash_helper(
            normal,
            gammadash1,
            gammadash2,
            gammadash1dash1,
            gammadash1dash2,
            gammadash2dash2,
            nfp, cp_m, cp_n,
            net_poloidal_current_amperes,
            net_toroidal_current_amperes,
            quadpoints_phi,
            quadpoints_theta,
            stellsym,
            current_scale)
    # The Kx operator, acts on the current potential harmonics (Phi).
    # Shape: (n_phi_x, n_theta_x, 3, n_dof), (n_phi_x, n_theta_x, 3)
    # Contravariant basis of x
    grad1_x, grad2_x = grad_helper(gammadash1, gammadash2)
    # Unit normal for x

    # Operators that acts on the quadcoil vector.
    # Shape: (n_phi_x, n_theta_x, 3, n_dof+1)
    Kdash1_op_x = np.concatenate([
        Kdash1_sv_op_x,
        Kdash1_const_x[:, :, :, None]
    ], axis=-1)
    Kdash2_op_x = np.concatenate([
        Kdash2_sv_op_x,
        Kdash2_const_x[:, :, :, None]
    ], axis=-1)

        

    # The Kx operator, acts on the QUADCOIL vector (Phi/current_scale, 1).
    AK_x = AK_x_trimmed/current_scale
    K_x_op = np.concatenate([
        AK_x, bK_x_trimmed[:, :, :, None]
    ], axis=-1)

    ''' nabla_x cdot [pi_x K(y)] K(x) '''

    # divergence of the unit normal
    # Shape: (n_phi_x, n_theta_x)
    div_n_x = (
        np.sum(grad1_x * unitnormaldash1_x, axis=-1)
        + np.sum(grad2_x * unitnormaldash2_x, axis=-1)
    )

    ''' div_x pi_x '''
    # Shape: (n_phi_x, n_theta_x, 3)
    n_x_dot_grad_n_x = (
        np.sum(unitnormal_x * grad1_x, axis=-1)[:, :, None] * unitnormaldash1_x
        + np.sum(unitnormal_x * grad2_x, axis=-1)[:, :, None] * unitnormaldash2_x
    )
    # Shape: (n_phi_x, n_theta_x, 3)
    div_pi_x = -(
        div_n_x[:, :, None] * unitnormal_x
        + n_x_dot_grad_n_x
    )

    # Functions to integrate using the single and double layer 
    # Laplacian kernels
    # 1e-7 is mu0/4pi
    integrand_single = 1e-7 * (
        # Term 1
        # n(x) div n K(x) 
        # - (
        #     grad phi partial_phi 
        #     + grad theta partial_theta
        # ) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        (
            unitnormal_x[:, :, :, None, None] * div_n_x[:, :, None, None, None] * K_x_op[:, :, None, :, :]
        ) 
        - (
            grad1_x[:, :, :, None, None] * Kdash1_op_x[:, :, None, :, :]
            + grad2_x[:, :, :, None, None] * Kdash2_op_x[:, :, None, :, :]
        ) 
        # Term 3
        # K(x) div pi_x 
        # + partial_phi K(x) grad phi
        # + partial_theta K(x) grad theta
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        + (K_x_op[:, :, :, None, :] * div_pi_x[:, :, None, :, None]) 
        + (
            Kdash1_op_x[:, :, :, None, :] * grad1_x[:, :, None, :, None]
            + Kdash2_op_x[:, :, :, None, :] * grad2_x[:, :, None, :, None]
        )
    ) 

    integrand_double = 1e-7 * (
        # Term 2
        # n(x) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x)) 
        (unitnormal_x[:, :, :, None, None] * K_x_op[:, :, None, :, :]) 
        # Term 4
        # K(x) n(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        - (K_x_op[:, :, :, None, :] * unitnormal_x[:, :, None, :, None])
    )

    return(
        K_x_op,
        integrand_single,
        integrand_double
    )

# Calculates the self-force operator.
def self_force_cylindrical(cp: CurrentPotentialFourier, current_scale, normalize=True):
    winding_surface = cp.winding_surface
    nfp = cp.nfp
    len_phi_1fp = len(winding_surface.quadpoints_phi)//nfp
    AK, bK = AK_helper(cp)
    AK_trimmed = AK[:len_phi_1fp]
    bK_trimmed = bK[:len_phi_1fp]
    (
        K_x_op_xyz,
        integrand_single_xyz,
        integrand_double_xyz
    ) = self_force_integrands_xyz(
        normal=winding_surface.normal()[:len_phi_1fp],
        unitnormal=winding_surface.unitnormal()[:len_phi_1fp],
        gammadash1=winding_surface.gammadash1()[:len_phi_1fp],
        gammadash2=winding_surface.gammadash2()[:len_phi_1fp],
        gammadash1dash1=winding_surface.gammadash1dash1()[:len_phi_1fp],
        gammadash1dash2=winding_surface.gammadash1dash2()[:len_phi_1fp],
        gammadash2dash2=winding_surface.gammadash2dash2()[:len_phi_1fp],
        nfp=nfp, 
        cp_m=cp.m, 
        cp_n=cp.n,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        quadpoints_phi=winding_surface.quadpoints_phi[:len_phi_1fp],
        quadpoints_theta=winding_surface.quadpoints_theta[:len_phi_1fp],
        stellsym=winding_surface.stellsym,
        AK_x_trimmed=AK_trimmed, 
        bK_x_trimmed=bK_trimmed,
        current_scale=current_scale
    )

    gamma_1fp = winding_surface.gamma()[:len_phi_1fp]
    # We must perform the xyz -> R,Phi,Z coordinate change twice for both
    # axis 2 and 3. Otherwise the operator will not have 
    # the same nfp-fold discrete symmetry as the equilibrium. 
    integrand_single_cylindrical = project_arr_cylindrical(
        gamma_1fp, 
        integrand_single_xyz
    )
    integrand_double_cylindrical = project_arr_cylindrical(
        gamma_1fp, 
        integrand_double_xyz
    )
    # The projection function assumes that the first 3 components of the array represents the 
    # phi, theta grid and resulting components of the array. Hence the swapaxes.
    integrand_single_cylindrical = project_arr_cylindrical(
        gamma_1fp, 
        integrand_single_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)
    integrand_double_cylindrical = project_arr_cylindrical(
        gamma_1fp, 
        integrand_double_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)

    K_op_cylindrical = project_arr_cylindrical(
        gamma_1fp, 
        K_x_op_xyz
    )
    # Performing the singular integral using BIEST
    integrand_single_cylindrical_reshaped = integrand_single_cylindrical.reshape((
        integrand_single_cylindrical.shape[0],
        integrand_single_cylindrical.shape[1],
        -1
    ))
    integrand_double_cylindrical_reshaped = integrand_double_cylindrical.reshape((
        integrand_double_cylindrical.shape[0],
        integrand_double_cylindrical.shape[1],
        -1
    ))
    result_single = np.zeros_like(integrand_single_cylindrical_reshaped)
    result_double = np.zeros_like(integrand_double_cylindrical_reshaped)
    biest_call.integrate_multi(
        gamma_1fp, # xt::pyarray<double> &gamma,
        integrand_single_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
        result_single, # xt::pyarray<double> &result,
        True,
        10, # int digits,
        nfp, # int nfp
    )
    biest_call.integrate_multi(
        gamma_1fp, # xt::pyarray<double> &gamma,
        integrand_double_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
        result_double, # xt::pyarray<double> &result,
        False,
        10, # int digits,
        nfp, # int nfp
    )
    # BIEST's convention has an extra 1/4pi.
    # We remove it now, and reshape the output 
    # into [n_phi(y), n_theta(y), 3(operates on K_y), 3, ndof].
    result_single = 4 * np.pi * result_single.reshape(
        result_single.shape[0],
        result_single.shape[1],
        3, 3, -1
    )
    # BIEST's convention has an extra 1/4pi.
    # We remove it now, and reshape the output 
    # into [n_phi(y), n_theta(y), 3(operates on K_y), 3, ndof].
    result_double = 4 * np.pi * result_double.reshape(
        result_double.shape[0],
        result_double.shape[1],
        3, 3, -1
    )
    # The final operator
    # [n_phi(y), n_theta(y), 3, ndof]
    # The negative sign is added by BIEST (NEED CONFIRMATION)
    self_force_cylindrical_operator = np.sum(
        K_op_cylindrical[:, :, :, None, :, None] 
        * (result_single - result_double)[:, :, :, :, None, :],
        axis = 2
    )
    
    if normalize:
        self_force_scale = avg_order_of_magnitude(self_force_cylindrical_operator)
        self_force_cylindrical_operator /= self_force_scale
    else:
        self_force_scale = 1
    return(self_force_cylindrical_operator, self_force_scale)

