import sys
sys.path.insert(1,'..')
import utils
from utils import avg_order_of_magnitude
from utils import sin_or_cos

# Import packages.
import numpy as np
from simsopt.field import CurrentPotentialFourier #, CurrentPotentialSolve
from operator_helper import norm_helper, Kdash_helper, diff_helper

def grid_curvature_operator(
    cp:CurrentPotentialFourier, 
    single_value_only:bool=True, 
    L2_unit=False,
    current_scale=1):
    ''' 
    Generates a (n_phi, n_theta, 3(xyz), dof, dof) bilinear 
    operator that calculates K cdot grad K on grid points
    specified by 
    cp.winding_surface.quadpoints_phi
    cp.winding_surface.quadpoints_theta
    from 2 sin/sin-cos Fourier current potentials 
    sin or cos(m theta - n phi) with m, n given by
    cp.m
    cp.n.

    Parameters: -----
    - `cp:CurrentPotential` - CurrentPotential object (Fourier representation)
    to optimize

    - ` free_GI:bool` - When `True`, treat I,G as free quantities as well. 
    the operator acts on the vector (Phi, I, G) rather than (Phi)

    Returns: -----
    A (n_phi, n_theta, 3(xyz), n_dof_phi, n_dof_phi) operator 
    that evaluates K dot grad K on grid points. When free_GI is True,
    the operator has shape (n_phi, n_theta, 3(xyz), n_dof_phi+2, n_dof_phi+2) 
    '''

    winding_surface = cp.winding_surface
    (
        Kdash1_sv_op, 
        Kdash2_sv_op, 
        Kdash1_const,
        Kdash2_const
    ) = Kdash_helper(cp, current_scale)
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = diff_helper(cp)
    
    ''' Pointwise product with partial r/partial phi or theta'''

    _, normN_prime_2d, inv_normN_prime_2d = norm_helper(winding_surface)
    
    term_a_op = (trig_diff_m_i_n_i@partial_phi)[:, :, None, :, None]\
        * Kdash2_sv_op[:, :, :, None, :]
    term_b_op = (trig_diff_m_i_n_i@partial_theta)[:, :, None, :, None]\
        * Kdash1_sv_op[:, :, :, None, :]    
    # Includes only the single-valued contributions
    K_dot_grad_K_operator_sv = (term_a_op-term_b_op) * inv_normN_prime_2d[:, :, None, None, None]
    
    if single_value_only:
        K_dot_grad_K_operator = K_dot_grad_K_operator_sv
        K_dot_grad_K_const = np.nan
    else:
        # NOTE: direction phi or 1 is poloidal in simsopt.
        G = cp.net_poloidal_current_amperes*current_scale
        I = cp.net_toroidal_current_amperes*current_scale
            
        # Constant component of K dot grad K
        # Shape: (n_phi, n_theta, 3(xyz))
        K_dot_grad_K_const = inv_normN_prime_2d[:, :, None]*(G*Kdash2_const - I*Kdash1_const)
        
        # Component of K dot grad K linear in Phi
        # Shape: (n_phi, n_theta, 3(xyz), n_dof)
        K_dot_grad_K_linear = inv_normN_prime_2d[:, :, None, None]*(
            (trig_diff_m_i_n_i@partial_phi)[:, :, None, :]*Kdash2_const[:, :, :, None]
            +G*Kdash2_sv_op
            -(trig_diff_m_i_n_i@partial_theta)[:, :, None, :]*Kdash1_const[:, :, :, None]
            -I*Kdash1_sv_op
        )
        
        K_dot_grad_K_operator = np.zeros((
            K_dot_grad_K_operator_sv.shape[0],
            K_dot_grad_K_operator_sv.shape[1],
            K_dot_grad_K_operator_sv.shape[2],
            K_dot_grad_K_operator_sv.shape[3]+1,
            K_dot_grad_K_operator_sv.shape[4]+1
        ))
        
        K_dot_grad_K_operator[:,:,:,:-1,:-1] = K_dot_grad_K_operator_sv
        K_dot_grad_K_operator[:,:,:,-1,:-1] = K_dot_grad_K_linear
        K_dot_grad_K_operator[:,:,:,-1,-1] = K_dot_grad_K_const
    
    # K cdot grad K is (term_a_op-term_b_op)/|N|. 
    # Shape: (n_phi, n_theta, 3(xyz), dof, dof)
    # Multiply by sqrt grid spacing, so that it's L2 norm^2 is K's surface integral.
    if L2_unit:
        # Here, 1/sqrt(|N|) cancels with the Jacobian |N|
        # whenever the square of K dot grad K is integrated.
        dtheta_coil = (winding_surface.quadpoints_theta[1] - winding_surface.quadpoints_theta[0])
        dphi_coil = (winding_surface.quadpoints_phi[1] - winding_surface.quadpoints_phi[0])
        L2_scale = np.sqrt(
            dphi_coil * dtheta_coil * normN_prime_2d[:, :, None, None, None]
        )
        K_dot_grad_K_operator *= L2_scale
    
    # Symmetrize:
    # We only care about symmetric Phi bar bar.
    K_dot_grad_K_operator = (K_dot_grad_K_operator+np.swapaxes(K_dot_grad_K_operator,3,4))/2

    return(K_dot_grad_K_operator/current_scale**2)

def grid_curvature_operator_pol_n_binorm(
    cp:CurrentPotentialFourier, 
    current_scale,
    single_value_only:bool=True):
    '''
    Calculates curvature component along the 
    poloidal (dr/dtheta)
    normal (n)
    and binormal (dr/dtheta x n)
    directions
    '''
    ws_gammadash2 = cp.winding_surface.gammadash2() # Partial r partial theta
    ws_normal = cp.winding_surface.normal()
    ws_gammadash2_unit = ws_gammadash2/np.linalg.norm(ws_gammadash2, axis=2)[:,:,None]
    ws_normal_unit = ws_normal/np.linalg.norm(ws_normal, axis=2)[:,:,None]
    binorm_unit = np.cross(
        ws_gammadash2_unit,
        ws_normal_unit,
    )
    return(
        grid_curvature_operator_project(
            cp, 
            unit1=ws_gammadash2_unit, 
            unit2=ws_normal_unit, 
            unit3=binorm_unit,
            current_scale=current_scale,
            one_field_period=True,
            single_value_only=single_value_only,
            L2_unit=False,
        )
    )

def grid_curvature_operator_cylindrical(
        cp:CurrentPotentialFourier, 
        current_scale,
        single_value_only:bool=False,
        normalize=True
    ):
    '''
    K dot nabla K components in R, Phi, Z. Only 1 field period.
    '''
    K_dot_grad_K = grid_curvature_operator(
        cp=cp, 
        single_value_only=single_value_only, 
        current_scale=current_scale,
        L2_unit=False
    )
    out = utils.project_field_operator_cylindrical(
        cp=cp, 
        operator=K_dot_grad_K,
    )
    if normalize:
        out_scale = avg_order_of_magnitude(out)
        out /= out_scale
    else:
        out_scale = 1
    return(out, out_scale)
    
