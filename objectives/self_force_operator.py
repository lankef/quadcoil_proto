import sys
import time
sys.path.insert(1,'..')
from utils import project_arr_cylindrical
# Import packages.
import numpy as np
from simsopt.field import CurrentPotentialFourier
from simsopt.geo import SurfaceRZFourier
from operator_helper import (
    norm_helper, Kdash_helper, 
    unitnormaldash, grad_helper
)
from f_b_and_k_operators import AK_helper
sys.path.insert(0, '../build')
import biest_call

def self_force_integrand_nominator_cylindrical(
        cp_x:CurrentPotentialFourier, current_scale, 
    ):
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
    # cp = cp_test_terms
    winding_surface_x = cp_x.winding_surface

    unitnormaldash1_x, unitnormaldash2_x = unitnormaldash(winding_surface_x)
    # An assortment of useful quantities related to K
    (
        Kdash1_sv_op_x, 
        Kdash2_sv_op_x, 
        Kdash1_const_x,
        Kdash2_const_x
    ) = Kdash_helper(cp_x, current_scale)

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

    # Contravariant basis of x
    grad1_x, grad2_x = grad_helper(winding_surface_x)
    # Unit normal for x
    unitnormal_x = winding_surface_x.unitnormal()
        
    # The Kx operator, acts on the current potential harmonics (Phi).
    # Shape: (n_phi_x, n_theta_x, 3, n_dof), (n_phi_x, n_theta_x, 3)
    AK_x, bK_x = AK_helper(cp_x)
    # The Kx operator, acts on the QUADCOIL vector (Phi/current_scale, 1).
    AK_x = AK_x/current_scale
    K_x_op = np.concatenate([
        AK_x, bK_x[:, :, :, None]
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
    # print('div_pi_x', div_pi_x)
    ''' 
    Integrands

    Integrands in the expression are sorted by their kernel and summed.
    '''
    # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz, resulting components), n_dof+1(x))
    mu0_over_4pi = 1e-7 
    func_single_xyz = mu0_over_4pi * (
        (
            unitnormal_x[:, :, :, None, None] 
            * div_n_x[:, :, None, None, None] 
            * K_x_op[:, :, None, :, :]
        ) # Term 1a
        + (
            - grad1_x[:, :, :, None, None] * Kdash1_op_x[:, :, None, :, :]
            - grad2_x[:, :, :, None, None] * Kdash2_op_x[:, :, None, :, :]
        ) # Term 1b
        + (
            K_x_op[:, :, :, None, :] * div_pi_x[:, :, None, :, None]
        ) # Term 3a
        + (
            Kdash1_op_x[:, :, :, None, :] * grad1_x[:, :, None, :, None]
            + Kdash2_op_x[:, :, :, None, :] * grad2_x[:, :, None, :, None]
        ) # Term 3b
    ) 
    # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz, resulting components), n_dof+1(x))
    func_double_xyz = mu0_over_4pi * (
        (unitnormal_x[:, :, :, None, None] * K_x_op[:, :, None, :, :]) # Term 2
        + (-K_x_op[:, :, :, None, :] * unitnormal_x[:, :, None, :, None]) # Term 4
    )

    # We must perform the xyz -> R,Phi,Z coordinate change twice for both
    # axis 2 and 3. Otherwise the operator will not have 
    # the same nfp-fold discrete symmetry as the equilibrium. 
    func_single_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        func_single_xyz
    )
    func_double_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        func_double_xyz
    )
    # The projection function assumes that the first 3 components of the array represents the 
    # phi, theta grid and resulting components of the array. Hence the swapaxes.
    func_single_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        func_single_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)
    func_double_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        func_double_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)

    K_op_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        K_x_op
    )
    # Shape: (n_phi_x, n_theta_x, 3(R, Phi Z, acts on K(y)), 3(R, Phi, Z, resulting components), n_dof+1(x))
    return(
        func_single_cylindrical, func_double_cylindrical,
        K_op_cylindrical,
    )

def self_force_operator_cylindrical(
        cp_x:CurrentPotentialFourier, current_scale):
    '''
    First, we need a temporary cp, cp_eval with a cp_eval.winding_surface
    containing only 1 field period to reduce array sizes by nfp times.
    BIEST seems to only support evaluating int f(x, y)da(x) 
    using the same grid for x and y. If this isn't the case, then 
    we need to have different CurrentPotential and Surface instances 
    for x and y.
    '''
    # Convert the surface type
    winding_surface_original_RZ = cp_x.winding_surface.to_RZFourier()
    len_phi = len(winding_surface_original_RZ.quadpoints_phi)
    len_theta = len(winding_surface_original_RZ.quadpoints_theta)

    # We assume the original winding_surface contains all field periods.
    # Copy the surface and limits the 
    winding_surface_1fp = SurfaceRZFourier( 
        nfp=winding_surface_original_RZ.nfp,
        stellsym=winding_surface_original_RZ.stellsym,
        mpol=winding_surface_original_RZ.mpol,
        ntor=winding_surface_original_RZ.ntor,
        # quadpoints_phi=np.arange(nzeta_coil)/nzeta_coil/3,
        quadpoints_phi=np.linspace(0, 1/cp_x.nfp, len_phi//cp_x.nfp, endpoint=False),
        # quadpoints_theta=np.arange(ntheta_coil)/ntheta_coil,
        quadpoints_theta=np.linspace(0, 1, len_theta, endpoint=False),
    )
    winding_surface_1fp.set_dofs(winding_surface_original_RZ.get_dofs())

    # Define a temporary CurrerntPotentialFourier object using this object
    cp_eval = CurrentPotentialFourier(
        winding_surface_1fp, mpol=cp_x.mpol, ntor=cp_x.ntor,
        net_poloidal_current_amperes=cp_x.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp_x.net_toroidal_current_amperes,
        stellsym=True
    )
    # No need to copy cp_x's dof because it'll not be used.
    
    (
        func_single_cylindrical, func_double_cylindrical,
        K_op_cylindrical,
    ) = self_force_integrand_nominator_cylindrical(cp_eval, current_scale)
    # Reshaping for using with BIEST
    func_single_cylindrical_reshaped = func_single_cylindrical.reshape((
        func_single_cylindrical.shape[0],
        func_single_cylindrical.shape[1],
        -1
    ))
    func_double_cylindrical_reshaped = func_double_cylindrical.reshape((
        func_double_cylindrical.shape[0],
        func_double_cylindrical.shape[1],
        -1
    ))
    # Mutation for the c++ function biest_call.integrate_multi
    result_single = np.zeros_like(func_single_cylindrical_reshaped)
    result_double = np.zeros_like(func_double_cylindrical_reshaped)

    # Calling BIEST, the singular integral code.
    time1 = time.time()
    biest_call.integrate_multi(
        cp_eval.winding_surface.gamma(), # xt::pyarray<double> &gamma,
        func_single_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
        result_single, # xt::pyarray<double> &result,
        True, # bool, whether to use the single layer kernel
        10, # int digits,
        cp_eval.nfp, # int nfp
        True # Whether to detect sign flip for a double-layer kernel.  
    )
    biest_call.integrate_multi(
        cp_eval.winding_surface.gamma(), # xt::pyarray<double> &gamma,
        func_double_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
        result_double, # xt::pyarray<double> &result,
        False, # bool, whether to use the single layer kernel
        10, # int digits,
        cp_eval.nfp, # int nfp
        True # Whether to detect sign flip for a double-layer kernel.  
    )
    time2 = time.time()
    num_integral = func_double_cylindrical_reshaped.shape[-1]*2# The total number of BIEST integrals evaluated
    print('BIEST run completed.')
    print('Total time(s):               ', time2-time1)
    print('Total number of integrals:   ', num_integral)
    print('Average time per integral(s):', (time2-time1)/num_integral)
    result_tot = (result_single + result_double).reshape(
        result_single.shape[0],
        result_single.shape[1],
        3, 3, -1
    )
    self_force_operator = np.sum(
        K_op_cylindrical[:, :, :, None, :, None] 
        * result_tot[:, :, :, :, None, :],
        axis = 2
    )
    return(self_force_operator)
