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
        quadpoints_phi_x, quadpoints_theta_x
    ):
    ''' 
    Calculates sheet current self-force based on Robin, Volpe 2022.
    '''
    cpst_y = cpst
    winding_surface_y = cpst_y.winding_surface
    cp_y = cpst_y.current_potential
    # Define the integration surface
    winding_surface_x = SurfaceRZFourier(
        nfp=winding_surface_y.nfp, 
        stellsym=winding_surface_y.stellsym, 
        mpol=winding_surface_y.mpol, 
        ntor=winding_surface_y.ntor, 
        quadpoints_phi=quadpoints_phi_x,
        quadpoints_theta=quadpoints_theta_x,
    )
    winding_surface_x.set_dofs(winding_surface_y.get_dofs())
    cp_x = CurrentPotentialFourier(
        winding_surface_x, 
        mpol=cp_y.mpol, 
        ntor=cp_y.ntor,
        net_poloidal_current_amperes=cp_y.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp_y.net_toroidal_current_amperes,
        stellsym=True
    )
    cpst_x = CurrentPotentialSolve(cp_x, cpst_y.plasma_surface, cpst_y.Bnormal_plasma)
    ''' Calculating necessary quantities '''
    (
        normal_x,
        normN_prime_2d_x,
        inv_normN_prime_2d_x
    ) = norm_helper(winding_surface_x)
    unitnormal_x = winding_surface_x.unitnormal()
    # The self-force contains an integral over phi, theta.
    # Refer to the coordinate being integrated over as 
    # phi_x, zeta_x, and the coordinate where the self-force
    # is evaluated as phi_y, zeta_y. 
    # We use the following shape convention for the integrands:
    # (n_phi_x, n_theta_x, n_phi_y, n_theta_y, 3(cartesian), n_dof+1(x), n_dof+1(y))
    # The vector (y - x)
    ymx = (
        winding_surface_y.gamma()[:, :, None, None, :]
        - winding_surface_x.gamma()[None, None, :, :, :]
    )
    # 1/|y - x|
    # Shape: (n_phi_x, n_theta_x)
    ymx_inv = 1/np.linalg.norm(ymx, axis=-1)
    # 1/|y - x|^3
    # Shape: (n_phi_x, n_theta_x)
    ymx_inv3 = ymx_inv**3
    # Partial derivatives of the unit normal n
    # Shape: (n_phi_x, n_theta_x, 3)
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
    ])
    Kdash2_op_x = np.concatenate([
        Kdash2_sv_op_x,
        Kdash2_const_x[:, :, :, None]
    ])
  
    # Contravariant basis of x
    grad1_x, grad2_x = grad_helper(winding_surface_x)

    # The Ky operator, acts on the current potential harmonics (Phi).
    # Shape: (n_phi_y, n_theta_y, 3, n_dof), (n_phi_y, n_theta_y, 3)
    AK_y, bK_y = AK_helper(cpst_y)
    # The Ky operator, acts on the QUADCOIL vector (Phi/current_scale, 1).
    AK_y = AK_y/current_scale
    K_y_op = np.concatenate([
        AK_y, bK_y[:, :, :, None]
    ], axis=-1)
        
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
    # grad_x n_x dot K_y
    # Operates on the single-valued component of the current potential.
    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, 3, n_dof+1)
    grad_n_x_dot_K_y_op = (
        np.sum(
            K_y_op[None, None, :, :, :, :] * unitnormaldash1_x[:, :, None, None, :, None], 
            axis=4
        ) * grad1_x[:, :, None, None, :, None]
        + np.sum(
            K_y_op[None, None, :, :, :, :] * unitnormaldash2_x[:, :, None, None, :, None], 
            axis=4
        ) * grad2_x[:, :, None, None, :, None]
    )
    # n_x dot K_y
    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, n_dof+1)
    n_x_dot_K_y_op = np.sum(
        (
            unitnormal_x[:, :, None, None, :, None] 
            * K_y_op[None, None, :, :, :, :]
        ),
        axis=4
    )
    # The linear term div_x (pi_x K_y) of integrand 1.
    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, n_dof+1(y))
    div_pi_x_K_y_op = -(
        div_n_x[:, :, None, None, None] * n_x_dot_K_y_op
        + np.sum(
            unitnormal_x[:, :, None, None, :, None] * grad_n_x_dot_K_y_op, 
            axis=4
        )
    )

    ''' pi_x K(y) cdot nabla_x K(x) '''

    # pi_x K_y
    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, 3, n_dof+1(y))
    pi_x_K_y_op = (
        K_y_op[None, None, :, :, :, :] 
        - n_x_dot_K_y_op[:, :, :, :, None, :] * unitnormal_x[:, :, None, None, :, None]
    )

    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, 3, n_dof+1(x), n_dof+1(y))
    pi_x_K_y_dot_grad_K_x = (
        (
            np.sum(pi_x_K_y_op[:, :, :, :, :, None, :] * grad1_x[:, :, None, None, :, None, None], axis=4) 
            * Kdash1_op_x[:, :, None, None, :, :, None]
        )
        + (
            np.sum(pi_x_K_y_op[:, :, :, :, :, None, :] * grad2_x[:, :, None, None, :, None], axis=4) 
            * Kdash2_op_x[:, :, None, None, :, :, None]
        )
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

    ''' div_x (K_x dot K_y) '''

    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, 3, n_dof+1(x), n_dof+1(y))
    div_K_x_dot_K_y = (
        np.sum(
            K_y_op[None, None, :, :, :, None, :] * Kdash1_op_x[:, :, None, None, :, :, None],
            axis=4
        )[:, :, :, :, None, :, :] * grad1_x[:, :, None, None, :, None, None]
        + np.sum(
            K_y_op[None, None, :, :, :, None, :] * Kdash2_op_x[:, :, None, None, :, :, None],
            axis=4
        )[:, :, :, :, None, :, :] * grad2_x[:, :, None, None, :, None, None]
    )

    ''' K_x dot K_y '''
    
    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, n_dof+1(x), n_dof+1(y))
    K_x_dot_K_y = np.sum(
        K_x_op[:, :, None, None, :, :, None] * K_y_op[None, None, :, :, :, None, :],
        axis=4
    )

    ''' (y-x) dot n_x '''
    ymx_dot_nx = np.sum(
        ymx * unitnormal_x[:, :, None, None, :], axis=-1
    )

    ''' Integrand 1 '''

    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, 3(cartesian), n_dof+1(x), n_dof+1(y))
    integrand_1_nominator = -(
        div_pi_x_K_y_op[:, :, :, :, None, None, :] * K_x_op[:, :, None, None, :, :, None]
        + pi_x_K_y_dot_grad_K_x
    )

    ''' Integrand 2 '''

    # Shape: (n_phi_x, n_theta_x, n_phi_y, n_theta_y, 3(cartesian), n_dof+1(x), n_dof+1(y))
    integrand_2_nominator = (
        n_x_dot_K_y_op[:, :, :, :, None, None, :]
        * ymx_dot_nx[:, :, :, :, None, None, None]
        * K_x_op[:, :, None, None, :, :, None]
    )

    ''' Integrand 3 '''

    integrand_3_nominator = (
        K_x_dot_K_y[:, :, :, :, None, :, :] * div_pi_x[:, :, None, None, :, None, None]
        + div_K_x_dot_K_y
    )

    ''' Integrand 4 '''

    integrand_4_nominator = -(
        K_x_dot_K_y[:, :, :, :, None, :, :]
        * ymx_dot_nx[:, :, :, :, None, None, None]
        * unitnormal_x[:, :, None, None, :, None, None]
    )

    return(
        cpst_x,
        ymx_inv,
        ymx_inv3,
        integrand_1_nominator,
        integrand_2_nominator,
        integrand_3_nominator,
        integrand_4_nominator,
    )

    