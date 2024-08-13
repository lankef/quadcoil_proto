# Import packages.
import numpy as np
import sys
sys.path.insert(1,'..')
import utils
from utils import avg_order_of_magnitude, sin_or_cos
from simsopt.field import CurrentPotentialSolve, CurrentPotentialFourier
import simsoptpp as sopp
from operator_helper import diff_helper


def f_B_operator_and_current_scale(cpst: CurrentPotentialSolve, normalize=True):
    '''
    Produces a dimensionless f_B and K operator that act on X by 
    tr(AX). Also produces a scaling factor current_scale that is an 
    estimate for the order of magnitude of Phi.

    -- Input: 
    A CurrentPotentialSolve.
    -- Output:
    1. An f_B operator with shape (nfod+1, ndof+1). Produces a 
    scalar when acted on X by tr(AX).
    2. f_B operator in REGCOIL
    3. A scalar scaling factor that estimates the order of magnitude
    of Phi. Used when constructing other operators, 
    such as grid_curvature_operator_cylindrical
    and K_operator_cylindrical.
    '''
    # Equivalent to A in regcoil.
    normN = np.linalg.norm(cpst.plasma_surface.normal().reshape(-1, 3), axis=-1)
    # The matrices may not have been precomputed
    try:
        B_normal = cpst.gj/np.sqrt(normN[:, None])
    except AttributeError:
        cpst.B_matrix_and_rhs()
        B_normal = cpst.gj/np.sqrt(normN[:, None])
    # Scaling factor to make X matrix dimensionless. 
    current_scale = avg_order_of_magnitude(B_normal)/avg_order_of_magnitude(cpst.b_e)
    ''' f_B operator '''
    # Scaling blocks of the operator
    B_normal_scaled = B_normal/current_scale
    ATA_scaled = B_normal_scaled.T@B_normal_scaled
    ATb_scaled = B_normal_scaled.T@cpst.b_e
    bTb_scaled = np.dot(cpst.b_e,cpst.b_e)
    # Concatenating blocks into the operator
    f_B_x_operator = np.block([
        [ATA_scaled, -ATb_scaled[:, None]],
        [-ATb_scaled[None, :], bTb_scaled[None, None]]
    ])/2*cpst.current_potential.nfp
    if normalize:
        f_B_scale = avg_order_of_magnitude(f_B_x_operator)
        f_B_x_operator /= f_B_scale
    else:
        f_B_scale = 1
    return(f_B_x_operator, B_normal, current_scale, f_B_scale)

def K_operator_cylindrical(cp: CurrentPotentialFourier, current_scale, normalize=True):
    '''
    Produces a dimensionless K operator that act on X by 
    tr(AX). Note that this operator is linear in Phi, rather
    than X.

    The K oeprator has shape (#grid per period, 3, nfod+1, ndof+1). 
    tr(A[i, j, :, :]X) cannot gives the grid value of a K component
    in (R, phi, Z). 

    It cannot directly produce a scalar objective, but can be used 
    for constructing norms (L1, L2). 
    '''
    AK_operator, _ = K_operator(
        cp=cp, 
        current_scale=current_scale, 
        normalize=False
    )
    AK_operator_cylindrical = utils.project_field_operator_cylindrical(
        cp=cpst.current_potential,
        operator=AK_operator
    )
    if normalize:
        AK_scale = avg_order_of_magnitude(AK_operator_cylindrical)
        AK_operator_cylindrical /= AK_scale
    return(AK_operator_cylindrical, AK_scale)

def AK_helper(cp: CurrentPotentialFourier):
    '''
    We take advantage of the fj matrix already 
    implemented in CurrentPotentialSolve to calculate K.
    This is a helper method that applies the necessary units 
    and scaling factors. 
    
    When L2_unit=True, the resulting matrices 
    contains the surface element and jacobian for integrating K^2
    over the winding surface.

    When L2_unit=False, the resulting matrices calculates
    the actual components of K.
    '''
    winding_surface = cp.winding_surface
    (
        _, # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _, # partial_phi_phi,
        _, # partial_phi_theta,
        _, # partial_theta_theta,
    ) = diff_helper(cp)
    inv_normN_prime_2d = 1/np.linalg.norm(winding_surface.normal(), axis=-1)
    dg1 = winding_surface.gammadash1()
    dg2 = winding_surface.gammadash2()
    G = cp.net_poloidal_current_amperes
    I = cp.net_toroidal_current_amperes
    AK = inv_normN_prime_2d[:, :, None, None] * (
        dg2[:, :, :, None] * (trig_diff_m_i_n_i @ partial_phi)[:, :, None, :]
        - dg1[:, :, :, None] * (trig_diff_m_i_n_i @ partial_theta)[:, :, None, :]
    )
    bK = inv_normN_prime_2d[:, :, None] * (
        dg2 * G
        - dg1 * I
    )
    return(AK, bK)
    
def AK_helper_legacy(cpst: CurrentPotentialSolve):
    # Uses the fj matrix in cpst to calculate the
    # K operator. May have some inconsistencies as of 
    # Aug 2 2024 in robin_volpe.ipynb
    signed_fj = -cpst.fj
    signed_d = cpst.d
    dzeta_coil = (
        cpst.winding_surface.quadpoints_phi[1] 
        - cpst.winding_surface.quadpoints_phi[0]
    )
    dtheta_coil = (
        cpst.winding_surface.quadpoints_theta[1] 
        - cpst.winding_surface.quadpoints_theta[0]
    )
    normal_vec = cpst.winding_surface.normal()
    normn = np.sqrt(np.sum(normal_vec**2, axis=-1)) # |N|
    normn = normn.reshape(-1)
    factor = (np.sqrt(dzeta_coil * dtheta_coil) * normn)**-1
    signed_fj = signed_fj * factor[:, None, None]
    signed_d = signed_d * factor[:, None]
    # test_K_2 = (
    #     -(cpst.fj @ cp.get_dofs() - cpst.d) 
    #     / 
    # )
    shape_gamma = (
        len(cpst.winding_surface.quadpoints_phi),
        len(cpst.winding_surface.quadpoints_theta),
        3
    )
    AK = signed_fj.reshape(
        # Reshape to the same shape as 
        list(shape_gamma)+[-1] 
    )
    bK = signed_d.reshape(
        # Reshape to the same shape as 
        shape_gamma
    )
    return(AK, bK)

def K_operator(cp: CurrentPotentialFourier, current_scale, normalize=True):
    '''
    Produces a dimensionless K operator that act on X by 
    tr(AX). Note that this operator is linear in Phi, rather
    than X.

    The K operator has shape:
    (n_phi, n_theta, 3, nfod+1, ndof+1). 
    tr(A[i, j, :, :]X) cannot gives the grid value of a K component
    in (R, phi, Z). 

    It cannot directly produce a scalar objective, but can be used 
    for constructing norms (L1, L2). 
    '''
    AK, bK = AK_helper(cp)
    # To fill the part of ther operator representing
    # 2nd order coefficients
    AK_blank_square = np.zeros(
        list(AK.shape)+[AK.shape[-1]]
    )
    AK_operator = np.block([
        [AK_blank_square,                     AK[:, :, :, :, None]/current_scale],
        [np.zeros_like(AK)[:, :, :, None, :], bK[:, :, :, None, None]]
    ])
    AK_operator = (AK_operator+np.swapaxes(AK_operator, -2, -1))/2
    if normalize:
        AK_scale = avg_order_of_magnitude(AK_operator)
        AK_operator /= AK_scale
    return(AK_operator, AK_scale)

def K_l2_operator(cp: CurrentPotentialFourier, current_scale, normalize=True):
    '''
    An operator that calculates the L2 norm of K.
    Shape: (n_phi (1 field period) x n_theta, n_dof+1, n_dof+1)
    '''
    AK, bK = AK_helper(cp)
    AK = AK[
        :AK.shape[0]//cp.nfp,
        :,
        :,
        :,
    ]
    # Take only one field period
    bK = bK[
        :bK.shape[0]//cp.nfp,
        :,
        :,
    ]
    # To fill the part of ther operator representing
    # 2nd order coefficients
    AK_scaled = (AK/current_scale)
    ATA_K_scaled = np.matmul(np.swapaxes(AK_scaled, -1, -2),AK_scaled)
    ATb_K_scaled = np.sum(AK_scaled*bK[:,:, :, None], axis=-2)
    bTb_K_scaled = np.sum(bK*bK, axis=-1)
    AK_l2_operator = np.block([
        [ATA_K_scaled, ATb_K_scaled[:, :, :, None]],
        [ATb_K_scaled[:, :, None, :], bTb_K_scaled[:, :, None, None]]
    ])
    AK_l2_operator = AK_l2_operator.reshape((
        -1, 
        AK_l2_operator.shape[-2],
        AK_l2_operator.shape[-1]
    ))
    if normalize:
        AK_l2_scale = avg_order_of_magnitude(AK_l2_operator)
        AK_l2_operator /= AK_l2_scale
    else:
        AK_l2_scale = 1
    return(
        AK_l2_operator, AK_l2_scale,
    )

def K_theta(cp, current_scale, normalize=True):
    ''' 
    K in the theta direction. Used to eliminate windowpane coils.  
    by K_theta >= -G. (Prevents current from flowing against G).
    '''
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
    partial_phi = -cp.n*trig_choice*iden*cp.nfp*2*np.pi
    phi_grid = np.pi*2*winding_surface.quadpoints_phi[:, None]
    theta_grid = np.pi*2*winding_surface.quadpoints_theta[None, :]
    trig_diff_m_i_n_i = sin_or_cos(
        (cp.m)[None, None, :]*theta_grid[:, :, None]
        -(cp.n*cp.nfp)[None, None, :]*phi_grid[:, :, None],
        -trig_choice
    )
    K_theta_shaped = (trig_diff_m_i_n_i@partial_phi)
    # Take 1 field period
    K_theta_shaped = K_theta_shaped[:K_theta_shaped.shape[0]//cp.nfp, :]
    K_theta = K_theta_shaped.reshape((-1, K_theta_shaped.shape[-1]))
    K_theta_scaled = K_theta/current_scale
    ATA_scaled = np.zeros((K_theta_scaled.shape[0], K_theta_scaled.shape[1],  K_theta_scaled.shape[1]))
    ATb_scaled = K_theta_scaled/2
    bTb_scaled = cp.net_poloidal_current_amperes*np.ones((K_theta_scaled.shape[0], 1, 1))
    # Concatenating blocks into the operator
    K_theta_operator = np.block([
        [ATA_scaled, ATb_scaled[:, :, None]],
        [ATb_scaled[:, None, :], bTb_scaled]
    ])
    if normalize:
        K_theta_scale = avg_order_of_magnitude(K_theta_operator)
        K_theta_operator /= K_theta_scale
    else:
        K_theta_scale = 1
    return(K_theta_operator, current_scale, K_theta_scale)

