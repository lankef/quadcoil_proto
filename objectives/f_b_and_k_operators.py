# Import packages.
import numpy as np
import utils
from utils import avg_order_of_magnitude, sin_or_cos
# from simsopt.objectives import SquaredFlux
# from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier
import simsoptpp as sopp
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

def K_operator_cylindrical(cpst: CurrentPotentialSolve, current_scale, normalize=True):
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
    # Equivalent to A in regcoil.
    normN = np.linalg.norm(cpst.plasma_surface.normal().reshape(-1, 3), axis=-1)
    B_normal = cpst.gj/np.sqrt(normN[:, None])
    # Scaling factor to make X matrix dimensionless. 
    current_scale = avg_order_of_magnitude(B_normal)/avg_order_of_magnitude(cpst.b_e)
    # fj's angles doesn't match cp.k()
    K_angle_scale = (np.pi*2)**2/cpst.plasma_surface.nfp
    AK = (-cpst.fj*K_angle_scale).reshape(
        # Reshape to the same shape as 
        list(cpst.winding_surface.gamma().shape)+[-1] 
    )
    bK = (cpst.d*K_angle_scale).reshape(
        # Reshape to the same shape as 
        cpst.winding_surface.gamma().shape
    )
    # To fill the part of ther operator representing
    # 2nd order coefficients
    AK_blank_square = np.zeros(
        list(AK.shape)+[AK.shape[-1]]
    )
    AK_operator = np.block([
        [AK_blank_square,                     AK[:, :, :, :, None]/current_scale],
        [np.zeros_like(AK)[:, :, :, None, :], bK[:, :, :, None, None]]
    ])
    AK_operator = 0.5*(AK_operator+np.swapaxes(AK_operator, -2, -1))
    AK_operator_cylindrical = utils.project_field_operator_cylindrical(
        cp=cpst.current_potential,
        operator=AK_operator
    )
    if normalize:
        AK_scale = avg_order_of_magnitude(AK_operator_cylindrical)
        AK_operator_cylindrical /= AK_scale
    return(AK_operator_cylindrical, AK_scale)

def K_l2_operator(cpst: CurrentPotentialSolve, current_scale, normalize=True, L2_unit=False):
    AK = (-cpst.fj).reshape(
        # Reshape to the same shape as 
        list(cpst.winding_surface.gamma().shape)+[-1] 
    )
    bK = (cpst.d).reshape(
        # Reshape to the same shape as 
        cpst.winding_surface.gamma().shape
    )
    contig = np.ascontiguousarray
    if cpst.winding_surface.stellsym:
        ndofs_half = cpst.current_potential.num_dofs()
    else:
        ndofs_half = cpst.current_potential.num_dofs() // 2
    m = cpst.current_potential.m[:ndofs_half]
    n = cpst.current_potential.n[:ndofs_half]
    phi_mesh, theta_mesh = np.meshgrid(
        cpst.winding_surface.quadpoints_phi, 
        cpst.winding_surface.quadpoints_theta, 
        indexing='ij'
    )
    zeta_coil = np.ravel(phi_mesh)
    theta_coil = np.ravel(theta_mesh)
    dr_dzeta = cpst.winding_surface.gammadash1().reshape(-1, 3)
    dr_dtheta = cpst.winding_surface.gammadash2().reshape(-1, 3)
    G = cpst.current_potential.net_poloidal_current_amperes
    I = cpst.current_potential.net_toroidal_current_amperes
    normal_coil = cpst.winding_surface.normal().reshape(-1, 3)
    bK, AK = sopp.winding_surface_field_K2_matrices(
        contig(dr_dzeta), contig(dr_dtheta), contig(normal_coil), cpst.winding_surface.stellsym,
        contig(zeta_coil), contig(theta_coil), cpst.ndofs, contig(m), contig(n), 
        cpst.winding_surface.nfp, G, I
    )
    bK = bK[:bK.shape[0]//cpst.winding_surface.nfp]
    AK = AK[:AK.shape[0]//cpst.winding_surface.nfp]
    if L2_unit:
        dzeta_coil = (
            cpst.winding_surface.quadpoints_phi[1] 
            - cpst.winding_surface.quadpoints_phi[0]
        )
        dtheta_coil = (
            cpst.winding_surface.quadpoints_theta[1] 
            - cpst.winding_surface.quadpoints_theta[0]
        )
        AK = AK * 2 * np.pi * np.sqrt(dzeta_coil * dtheta_coil)
        bK = bK * 2 * np.pi * np.sqrt(dzeta_coil * dtheta_coil)
    else:
        normN_prime = np.linalg.norm(cpst.winding_surface.normal(), axis=-1)
        # flaten
        normN_prime = normN_prime.flatten()
        normN_prime = normN_prime[:normN_prime.shape[0]//cpst.winding_surface.nfp]
        AK = AK/np.sqrt(normN_prime)[:, None, None]*(np.pi * 2)
        bK = bK/np.sqrt(normN_prime)[:, None]*(np.pi * 2)
    # To fill the part of ther operator representing
    # 2nd order coefficients
    AK_scaled = (AK/current_scale)
    ATA_K_scaled = np.matmul(np.swapaxes(AK_scaled, -1, -2),AK_scaled)
    ATb_K_scaled = np.sum(AK_scaled*bK[:,:,None], axis=-2)
    bTb_K_scaled = np.sum(bK*bK, axis=-1)
    AK_l2_operator = np.block([
        [ATA_K_scaled, -ATb_K_scaled[:, :, None]],
        [-ATb_K_scaled[:, None, :], bTb_K_scaled[:, None, None]]
    ])#.reshape((-1, n_dof+1, n_dof+1))
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

