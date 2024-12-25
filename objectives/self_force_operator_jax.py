import jax.numpy as jnp
from operator_helper import (
    Kdash_helper, 
    unitnormaldash, grad_helper
)
from f_b_and_k_operators import K_helper

import sys
sys.path.insert(0, '../build')
import biest_call
sys.path.insert(0, '..')
from jax import jit
from functools import partial
from utils import project_arr_cylindrical

# Calculates the integrands in Robin, Volpe from a number of arrays.
# The arrays needs trimming compared to the outputs
# with a standard cp.
# The inputs are array properties of a surface object
# containing only one field period so that the code is easy to port 
# into c++.
@partial(jit, static_argnames=[
    'quadpoints_phi',
    'quadpoints_theta',
    'nfp', 'cp_m', 'cp_n',
    'stellsym',
])
def self_force_integrands_xyz(
        normal,
        unitnormal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    ''' 
    Calculates the nominators of the sheet current self-force in Robin, Volpe 2022.
    The K_y dependence is lifted outside the integrals. Therefore, the nominator 
    is a linear function of phi. This function produces two arrays with shape
    
    integrand[n_phi_x, n_theta_x, 3(xyz, to act on Ky), 3(xyz), n_dof + 1]
    
    Here, the last slice of the last axis stores c, and the rest of the last axis stores b.
    This is to merge the single(double) layer laplacian integral in b and c in one 
    BIEST call.

    and the output from K_helper in the same concatenated form.
    ''' 
    # winding_surface = cp.winding_surface
    len_phi_1fp = len(quadpoints_phi)//nfp
    normal = normal[:len_phi_1fp]
    unitnormal = unitnormal[:len_phi_1fp]
    gammadash1 = gammadash1[:len_phi_1fp]
    gammadash2 = gammadash2[:len_phi_1fp]
    gammadash1dash1 = gammadash1dash1[:len_phi_1fp]
    gammadash1dash2 = gammadash1dash2[:len_phi_1fp]
    gammadash2dash2 = gammadash2dash2[:len_phi_1fp]
    nfp = nfp
    cp_m = cp_m
    cp_n = cp_n
    net_poloidal_current_amperes = net_poloidal_current_amperes
    net_toroidal_current_amperes = net_toroidal_current_amperes
    quadpoints_phi = quadpoints_phi[:len_phi_1fp]
    quadpoints_theta = quadpoints_theta
    stellsym = stellsym
    b_K, c_K = K_helper(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
    )
    b_K_trimmed = b_K[:len_phi_1fp]
    c_K_trimmed = c_K[:len_phi_1fp]

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
        b_Kdash1, 
        b_Kdash2, 
        c_Kdash1,
        c_Kdash2
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
        stellsym
    )
    # The Kx operator, acts on the current potential harmonics (Phi).
    # Shape: (n_phi_x, n_theta_x, 3, n_dof), (n_phi_x, n_theta_x, 3)
    # Contravariant basis of x
    grad1_x, grad2_x = grad_helper(gammadash1, gammadash2)
    # Unit normal for x
    # We temporarily concatenate b_K, c_K and b_Kdash, c_Kdash,
    # because they need to go through the same array operations.
    # Concatenating b and c will make it easier to check for typos.
    # The concatenated array is equivalent to an operator that 
    # acts on the vector (phi, 1).
    # Shape: (n_phi_x, n_theta_x, 3, n_dof+1)
    Kdash1_op_x = jnp.concatenate([
        b_Kdash1,
        c_Kdash1[:, :, :, None]
    ], axis=-1)
    Kdash2_op_x = jnp.concatenate([
        b_Kdash2,
        c_Kdash2[:, :, :, None]
    ], axis=-1)
    K_concat = jnp.concatenate([
        b_K_trimmed, c_K_trimmed[:, :, :, None]
    ], axis=-1)

    ''' nabla_x cdot [pi_x K(y)] K(x) '''

    # divergence of the unit normal
    # Shape: (n_phi_x, n_theta_x)
    div_n_x = (
        jnp.sum(grad1_x * unitnormaldash1_x, axis=-1)
        + jnp.sum(grad2_x * unitnormaldash2_x, axis=-1)
    )

    ''' div_x pi_x '''
    # Shape: (n_phi_x, n_theta_x, 3)
    n_x_dot_grad_n_x = (
        jnp.sum(unitnormal_x * grad1_x, axis=-1)[:, :, None] * unitnormaldash1_x
        + jnp.sum(unitnormal_x * grad2_x, axis=-1)[:, :, None] * unitnormaldash2_x
    )
    # Shape: (n_phi_x, n_theta_x, 3)
    div_pi_x = -(
        div_n_x[:, :, None] * unitnormal_x
        + n_x_dot_grad_n_x
    )

    # Functions to integrate using the single and double layer 
    # Laplacian kernels
    # 1e-7 is mu0/4pi
    integrand_single_concat = 1e-7 * (
        # Term 1
        # n(x) div n K(x) 
        # - (
        #     grad phi partial_phi 
        #     + grad theta partial_theta
        # ) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        (
            unitnormal_x[:, :, :, None, None] * div_n_x[:, :, None, None, None] * K_concat[:, :, None, :, :]
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
        + (K_concat[:, :, :, None, :] * div_pi_x[:, :, None, :, None]) 
        + (
            Kdash1_op_x[:, :, :, None, :] * grad1_x[:, :, None, :, None]
            + Kdash2_op_x[:, :, :, None, :] * grad2_x[:, :, None, :, None]
        )
    ) 

    integrand_double_concat = 1e-7 * (
        # Term 2
        # n(x) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x)) 
        (unitnormal_x[:, :, :, None, None] * K_concat[:, :, None, :, :]) 
        # Term 4
        # K(x) n(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        - (K_concat[:, :, :, None, :] * unitnormal_x[:, :, None, :, None])
    )
    return(
        integrand_single_concat,
        integrand_double_concat,
        K_concat,
    )

# Calculates the self-force operator.
def self_force_cylindrical_BIEST(
        normal,
        unitnormal,
        gamma,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
        skip_integral=False,
):
    nfp = nfp
    len_phi_1fp = len(quadpoints_phi)//nfp
    (
        integrand_single_concat,
        integrand_double_concat,
        K_concat,
    ) = self_force_integrands_xyz(
        normal=normal,
        unitnormal=unitnormal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=net_toroidal_current_amperes,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=nfp,
        cp_m=cp_m,
        cp_n=cp_n,
        stellsym=stellsym,
    )

    gamma_1fp = gamma[:len_phi_1fp]
    # We must perform the xyz -> R,Phi,Z coordinate change twice for both
    # axis 2 and 3. Otherwise the operator will not have 
    # the same nfp-fold discrete symmetry as the equilibrium. 
    integrand_single_concat_cylindrical = project_arr_cylindrical(gamma_1fp, integrand_single_concat)
    integrand_double_concat_cylindrical = project_arr_cylindrical(gamma_1fp, integrand_double_concat)
    # The projection function assumes that the first 3 components of the array represents the 
    # phi, theta grid and resulting components of the array. Hence the swapaxes.
    integrand_single_concat_cylindrical = project_arr_cylindrical(
        gamma_1fp, 
        integrand_single_concat_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)
    integrand_double_concat_cylindrical = project_arr_cylindrical(
        gamma_1fp, 
        integrand_double_concat_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)
    K_concat_cylindrical = project_arr_cylindrical(gamma_1fp, K_concat)

    if skip_integral:
        return(K_concat, integrand_single_concat_cylindrical, integrand_double_concat_cylindrical)

    # Performing the singular integral using BIEST
    integrand_single_concat_cylindrical_reshaped = integrand_single_concat_cylindrical.reshape((
        integrand_single_concat_cylindrical.shape[0],
        integrand_single_concat_cylindrical.shape[1],
        -1
    ))
    integrand_double_concat_cylindrical_reshaped = integrand_double_concat_cylindrical.reshape((
        integrand_double_concat_cylindrical.shape[0],
        integrand_double_concat_cylindrical.shape[1],
        -1
    ))
    result_single_concat = jnp.zeros_like(integrand_single_concat_cylindrical_reshaped)
    result_double_concat = jnp.zeros_like(integrand_double_concat_cylindrical_reshaped)
    biest_call.integrate_multi(
        gamma_1fp, # xt::pyarray<double> &gamma,
        integrand_single_concat_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
        result_single_concat, # xt::pyarray<double> &result,
        True,
        10, # int digits,
        nfp, # int nfp
    )
    biest_call.integrate_multi(
        gamma_1fp, # xt::pyarray<double> &gamma,
        integrand_double_concat_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
        result_double_concat, # xt::pyarray<double> &result,
        False,
        10, # int digits,
        nfp, # int nfp
    )
    # BIEST's convention has an extra 1/4pi.
    # We remove it now, and reshape the output 
    # into [n_phi(y), n_theta(y), 3(operates on K_y), 3, ndof+1].
    result_single_concat = 4 * jnp.pi * result_single_concat.reshape(
        result_single_concat.shape[0],
        result_single_concat.shape[1],
        3, 3, -1
    )
    # BIEST's convention has an extra 1/4pi.
    # We remove it now, and reshape the output 
    # into [n_phi(y), n_theta(y), 3(operates on K_y), 3, ndof+1].
    result_double_concat = 4 * jnp.pi * result_double_concat.reshape(
        result_double_concat.shape[0],
        result_double_concat.shape[1],
        3, 3, -1
    )
    # To simplify the math, we construct a symmetric [n_phi(y), n_theta(y), 3, ndof+1, ndof+1].
    # operator that acts on (phi, 1), 
    # then read off the individual blocks as A, b and c.
    self_force_cylindrical_O = jnp.sum(
        K_concat_cylindrical[:, :, :, None, :, None] 
        * (result_single_concat - result_double_concat)[:, :, :, :, None, :],
        axis = 2
    )
    self_force_cylindrical_O = (self_force_cylindrical_O + self_force_cylindrical_O.swapaxes(-1, -2))/2
    A_sf = self_force_cylindrical_O[:, :, :, :-1, :-1]
    b_sf = self_force_cylindrical_O[:, :, :, :-1, -1]
    c_sf = self_force_cylindrical_O[:, :, :, -1, -1]
    return(A_sf, b_sf, c_sf)

