# Import packages.
import numpy as np
import utils
from utils import avg_order_of_magnitude
# from simsopt.objectives import SquaredFlux
# from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier
# from simsoptpp import WindingSurfaceBn_REGCOIL
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
    ])
    if normalize:
        f_B_scale = avg_order_of_magnitude(f_B_x_operator)
        f_B_x_operator /= f_B_scale
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

def K_l2_operator(cpst: CurrentPotentialSolve, current_scale, normalize=True):
    AK = (-cpst.fj).reshape(
        # Reshape to the same shape as 
        list(cpst.winding_surface.gamma().shape)+[-1] 
    )
    bK = (cpst.d).reshape(
        # Reshape to the same shape as 
        cpst.winding_surface.gamma().shape
    )
    # To fill the part of ther operator representing
    # 2nd order coefficients
    AK_scaled = (AK/current_scale)
    ATA_K_scaled = np.matmul(np.swapaxes(AK_scaled, -1, -2),AK_scaled)
    ATb_K_scaled = np.sum(AK_scaled*bK[:,:,:,None], axis=-2)
    bTb_K_scaled = np.sum(bK*bK, axis=-1)

    AK_l2_operator = np.block([
        [ATA_K_scaled, ATb_K_scaled[:, :, :, None]],
        [ATb_K_scaled[:, :, None, :], bTb_K_scaled[:, :, None, None]]
    ])#.reshape((-1, n_dof+1, n_dof+1))
    
    if normalize:
        AK_l2_scale = avg_order_of_magnitude(AK_l2_operator)
        AK_l2_operator /= AK_l2_scale
    return(
        AK_l2_operator, AK_l2_scale,
    )
