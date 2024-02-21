import sys
sys.path.insert(1,'..')
import utils
from utils import avg_order_of_magnitude

# Import packages.
import numpy as np
from simsopt.field import CurrentPotentialFourier #, CurrentPotentialSolve

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
    # A helper method. When mode=0, calculates sin(x).
    # Otherwise calculates cos(x)
    sin_or_cos = lambda x, mode: np.where(mode==1, np.sin(x), np.cos(x))
    
    # The uniform index for phi contains first sin Fourier 
    # coefficients, then optionally cos is stellsym=False.
    n_harmonic = len(cp.m)
    iden = np.identity(n_harmonic)
    winding_surface = cp.winding_surface
    # When stellsym is enabled, Phi is a sin fourier series.
    # After a derivative, it becomes a cos fourier series.
    if winding_surface.stellsym:
        trig_choice = 1
    # Otherwise, it's a sin-cos series. After a derivative,
    # it becomes a cos-sin series.
    else:
        trig_choice = np.repeat([1,-1], n_harmonic//2)
    
    # Fourier 
    partial_theta = cp.m*trig_choice*iden*2*np.pi
    partial_phi = -cp.n*trig_choice*iden*cp.nfp*2*np.pi
    
    partial_theta_theta = -cp.m**2*iden*(2*np.pi)**2
    partial_phi_phi = -(cp.n*cp.nfp)**2*iden*(2*np.pi)**2
    partial_theta_phi = cp.n*cp.nfp*cp.m*iden*(2*np.pi)**2
    
    
    ''' Pointwise product with partial r/partial phi or theta'''
    # gammadash1() calculates partial r/partial phi, but phi in simsopt is from 0 to 1.
    # Shape: (n_phi, n_theta, 3(xyz))
    dg2 = winding_surface.gammadash2() # /(np.pi*2)
    dg1 = winding_surface.gammadash1() # /(np.pi*2*cp.nfp)
    
    dg22 = winding_surface.gammadash2dash2() # /(np.pi*2)**2
    dg11 = winding_surface.gammadash1dash1() # /(np.pi*2*cp.nfp)**2
    dg12 = winding_surface.gammadash1dash2() # /(np.pi*2*np.pi*2*cp.nfp)
    
    # Shape: (n_phi, n_theta)
    phi_grid = np.pi*2*winding_surface.quadpoints_phi[:, None]
    theta_grid = np.pi*2*winding_surface.quadpoints_theta[None, :]
    
    # Evaluate the value of partial phi and partial theta phi on these grid points
    
    # Inverse Fourier transform that transforms a dof 
    # array to grid values. trig_diff_m_i_n_i acts on 
    # odd-order derivatives of dof, where the sin coeffs 
    # become cos coefficients, and cos coeffs become
    # sin coeffs.
    # sin or sin-cos coeffs -> grid vals
    # Shape: (n_phi, n_theta, dof)
    trig_m_i_n_i = sin_or_cos(
        (cp.m)[None, None, :]*theta_grid[:, :, None]
        -(cp.n*cp.nfp)[None, None, :]*phi_grid[:, :, None],
        trig_choice
    )
    # cos or cos-sin coeffs -> grid vals
    # Shape: (n_phi, n_theta, dof)
    trig_diff_m_i_n_i = sin_or_cos(
        (cp.m)[None, None, :]*theta_grid[:, :, None]
        -(cp.n*cp.nfp)[None, None, :]*phi_grid[:, :, None],
        -trig_choice
    )
    

    # Length of the non-unit WS normal vector |N|,
    # its inverse (1/|N|) and its inverse's derivatives
    # w.r.t. phi(phi) and theta
    # Not to be confused with the normN (plasma surface Jacobian)
    # in Regcoil.
    normal_vec = winding_surface.normal()
    normN_prime_2d = np.sqrt(np.sum(normal_vec**2, axis=-1)) # |N|
    inv_normN_prime_2d = 1/normN_prime_2d # 1/|N|
    
    # Because Phi in simsopt is defined around the unit normal, rather 
    # than N, we need to calculate the derivative and double derivative 
    # of (dr/dtheta)/|N| and (dr/dphi)/|N|.
    # phi (phi) derivative of the normal's length
    normaldash1 = (
        np.cross(dg11, dg2)
        + np.cross(dg1, dg12)
    )
    # Theta derivative of the normal's length
    normaldash2 = (
        np.cross(dg12, dg2)
        + np.cross(dg1, dg22)
    )
    # Derivatives of 1/|N|:
    # d/dx(1/sqrt(f(x)^2 + g(x)^2 + h(x)^2)) 
    # = (-f(x)f'(x) - g(x)g'(x) - h(x)h'(x))
    # /(f(x)^2 + g(x)^2 + h(x)^2)^(3/2)
    denominator = np.sum(normal_vec**2, axis=-1)**1.5
    nominator_inv_normN_prime_2d_dash1 = -np.sum(normal_vec*normaldash1, axis=-1)
    nominator_inv_normN_prime_2d_dash2 = -np.sum(normal_vec*normaldash2, axis=-1)
    inv_normN_prime_2d_dash1 = nominator_inv_normN_prime_2d_dash1/denominator
    inv_normN_prime_2d_dash2 = nominator_inv_normN_prime_2d_dash2/denominator

    # Shape: (n_phi, n_theta, 3(xyz))
    # d[(1/|n|)(dgamma/dphi)]/dphi
    dg1_inv_n_dash1 = dg11*inv_normN_prime_2d[:,:,None] + dg1*inv_normN_prime_2d_dash1[:,:,None] 
    # d[(1/|n|)(dgamma/dphi)]/dtheta
    dg1_inv_n_dash2 = dg12*inv_normN_prime_2d[:,:,None] + dg1*inv_normN_prime_2d_dash2[:,:,None] 
    # d[(1/|n|)(dgamma/dtheta)]/dphi
    dg2_inv_n_dash1 = dg12*inv_normN_prime_2d[:,:,None] + dg2*inv_normN_prime_2d_dash1[:,:,None] 
    # d[(1/|n|)(dgamma/dtheta)]/dtheta
    dg2_inv_n_dash2 = dg22*inv_normN_prime_2d[:,:,None] + dg2*inv_normN_prime_2d_dash2[:,:,None] 
    
    # Operators that generates the derivative of K
    # Note the use of trig_diff_m_i_n_i for inverse
    # FT following odd-order derivatives.
    # Shape: (n_phi, n_theta, 3(xyz), n_dof)
    partial_theta_K_sv_op = (
        dg2_inv_n_dash2[:, :, None, :]
        *(trig_diff_m_i_n_i@partial_phi)[:, :, :, None]
        
        +cp.winding_surface.gammadash2()[:, :, None, :]
        *(trig_m_i_n_i@partial_theta_phi)[:, :, :, None]
        /normN_prime_2d[:, :, None, None]
        
        -dg1_inv_n_dash2[:, :, None, :]
        *(trig_diff_m_i_n_i@partial_theta)[:, :, :, None]
        
        -cp.winding_surface.gammadash1()[:, :, None, :]
        *(trig_m_i_n_i@partial_theta_theta)[:, :, :, None]
        /normN_prime_2d[:, :, None, None]
    )
    partial_theta_K_sv_op = np.swapaxes(partial_theta_K_sv_op, 2, 3)
    
    partial_phi_K_sv_op = (
        dg2_inv_n_dash1[:, :, None, :]
        *(trig_diff_m_i_n_i@partial_phi)[:, :, :, None]
        
        +cp.winding_surface.gammadash2()[:, :, None, :]
        *(trig_m_i_n_i@partial_phi_phi)[:, :, :, None]
        /normN_prime_2d[:, :, None, None]
        
        -dg1_inv_n_dash1[:, :, None, :]
        *(trig_diff_m_i_n_i@partial_theta)[:, :, :, None]
        
        -cp.winding_surface.gammadash1()[:, :, None, :]
        *(trig_m_i_n_i@partial_theta_phi)[:, :, :, None]
        /normN_prime_2d[:, :, None, None]
    )
    partial_phi_K_sv_op = np.swapaxes(partial_phi_K_sv_op, 2, 3)
    term_a_op = (trig_diff_m_i_n_i@partial_phi)[:, :, None, :, None]\
        * partial_theta_K_sv_op[:, :, :, None, :]
    term_b_op = (trig_diff_m_i_n_i@partial_theta)[:, :, None, :, None]\
        * partial_phi_K_sv_op[:, :, :, None, :]    
    # Includes only the single-valued contributions
    K_dot_grad_K_operator_sv = (term_a_op-term_b_op) * inv_normN_prime_2d[:, :, None, None, None]
    
    if single_value_only:
        K_dot_grad_K_operator = K_dot_grad_K_operator_sv
        K_dot_grad_K_const = np.nan
    else:
        # NOTE: direction phi or 1 is poloidal in simsopt.
        G = cp.net_poloidal_current_amperes*current_scale
        I = cp.net_toroidal_current_amperes*current_scale
        partial_phi_K_const = \
            dg2_inv_n_dash1*G \
            -dg1_inv_n_dash1*I
        partial_theta_K_const = \
            dg2_inv_n_dash2*G \
            -dg1_inv_n_dash2*I
            
        # Constant component of K dot grad K
        # Shape: (n_phi, n_theta, 3(xyz))
        K_dot_grad_K_const = inv_normN_prime_2d[:, :, None]*(G*partial_theta_K_const - I*partial_phi_K_const)
        
        # Component of K dot grad K linear in Phi
        # Shape: (n_phi, n_theta, 3(xyz), n_dof)
        K_dot_grad_K_linear = inv_normN_prime_2d[:, :, None, None]*(
            (trig_diff_m_i_n_i@partial_phi)[:, :, None, :]*partial_theta_K_const[:, :, :, None]
            +G*partial_theta_K_sv_op
            -(trig_diff_m_i_n_i@partial_theta)[:, :, None, :]*partial_phi_K_const[:, :, :, None]
            -I*partial_phi_K_sv_op
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

    # Not used in SDP.
    # # Calculating the operator's normal matrix
    # op_shape = K_dot_grad_K_operator.shape
    # K_dot_grad_K_operator_flat = K_dot_grad_K_operator.reshape(
    #     op_shape[0]*op_shape[1]*op_shape[2],
    #     op_shape[3]*op_shape[4]
    # )
    # normal_matrix = (K_dot_grad_K_operator_flat.T@K_dot_grad_K_operator_flat).reshape(
    #     (op_shape[4], op_shape[3], op_shape[3], op_shape[4])
    # )
    return(K_dot_grad_K_operator)

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
    
