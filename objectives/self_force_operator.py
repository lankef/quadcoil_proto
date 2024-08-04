import sys
sys.path.insert(1,'..')
import utils
from utils import avg_order_of_magnitude
from utils import sin_or_cos

# Import packages.
import numpy as np
from simsopt.field import CurrentPotentialFourier , CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier
from operator_helper import (
    norm_helper, Kdash_helper, 
    unitnormaldash, grad_helper
)
from f_b_and_k_operators import AK_helper

def self_force_integrand_nominator(
        cpst:CurrentPotentialSolve, current_scale, 
    ):
    ''' 
    Calculates the nominators of the sheet current self-force in Robin, Volpe 2022.
    The K_y dependence is lifted outside the integrals. Therefore, the nominator 
    this function calculates are operators that acts on the QUADCOIL vector
    (scaled Phi, 1). The operator produces a 
    (n_phi_x, n_theta_x, 3(cartesian, to act on Ky), 3(cartesian), n_dof+1)
    After the integral, this will become a (n_phi_y, n_theta_y, 3, 3, n_dof+1)
    tensor that acts on K(y) to produce a vector with shape (n_phi_y, n_theta_y, 3, n_dof+1, n_dof+1)
    Shape: (n_phi_x, n_theta_x, 3(cartesian), 3(cartesian), n_dof+1(x))
    '''
    cpst_x = cpst
    cp_x = cpst_x.current_potential
    winding_surface_x = cpst_x.winding_surface
    
    unitnormaldash1_x, unitnormaldash2_x = unitnormaldash(winding_surface_x)
    # An assortment of useful quantities related to K
    (
        Kdash1_sv_op_x, 
        Kdash2_sv_op_x, 
        Kdash1_const_x,
        Kdash2_const_x,
        _, # trig_m_i_n_i
        _, # trig_diff_m_i_n_i
        _, # partial_theta
        _, # partial_phi
        _, # partial_theta_theta
        _, # partial_phi_phi
        _, # partial_theta_phi
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
    AK_x, bK_x = AK_helper(cpst_x)
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
    div_pi_x = (
        div_n_x[:, :, None] * unitnormal_x
        + n_x_dot_grad_n_x
    )

    normN_prime_2d = np.ones_like(np.linalg.norm(winding_surface_x.normal(), axis=-1))

    ''' Term 1 '''
    # n(x) div n K(x) 
    # - (
    #     grad phi partial_phi 
    #     + grad theta partial_theta
    # ) K(x)
    # Shape: (n_phi_x, n_theta_x, 3(cartesian), 3(cartesian), n_dof+1(x))
    A_1 = normN_prime_2d[:, :, None, None, None] * (
        unitnormal_x[:, :, :, None, None] * div_n_x[:, :, None, None, None] * K_x_op[:, :, None, :, :]
        - grad1_x[:, :, :, None, None] * Kdash1_op_x[:, :, None, :, :]
        - grad2_x[:, :, :, None, None] * Kdash2_op_x[:, :, None, :, :]
    )

    ''' Term 2 '''
    # n(x) K(x)
    # Shape: (n_phi_x, n_theta_x, 3(cartesian), 3(cartesian), n_dof+1(x)) 
    A_2 = normN_prime_2d[:, :, None, None, None] * (
        unitnormal_x[:, :, :, None, None] * K_x_op[:, :, None, :, :]
    )

    ''' Term 3 '''
    # K(x) div pi_x 
    # + partial_phi K(x) grad phi
    # + partial_theta K(x) grad theta
    # Shape: (n_phi_x, n_theta_x, 3(cartesian), 3(cartesian), n_dof+1(x))
    A_3 = normN_prime_2d[:, :, None, None, None] * (
        K_x_op[:, :, :, None, :] * div_pi_x[:, :, None, :, None]
        + Kdash1_op_x[:, :, :, None, :] * grad1_x[:, :, None, :, None]
        + Kdash2_op_x[:, :, :, None, :] * grad2_x[:, :, None, :, None]
    )
    
    ''' Term 4 '''
    # K(x) n(x)
    # Shape: (n_phi_x, n_theta_x, 3(cartesian), 3(cartesian), n_dof+1(x))
    A_4 = normN_prime_2d[:, :, None, None, None] * (
        -K_x_op[:, :, :, None, :] * unitnormal_x[:, :, None, :, None]
    )
    
    return(
        A_1,
        A_2,
        A_3,
        A_4,
    )