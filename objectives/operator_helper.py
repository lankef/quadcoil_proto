import sys
sys.path.insert(1,'..')
import numpy as np
from utils import sin_or_cos
from simsopt.field import CurrentPotentialFourier

'''
This file includes some surface quantities that can 
potentially be reused by many objective functions,
such as K dot grad K and surface self force.
'''
def grad_helper(winding_surface):
    '''
    This is a helper method that calculates the contravariant 
    vectors, grad phi and grad theta, using the curvilinear coordinate 
    identities:
    - grad1: grad phi = (dg2 x (dg1 x dg2))/|dg1 x dg2|^2
    - grad2: grad theta = -(dg1 x (dg2 x dg1))/|dg1 x dg2|^2
    Shape: (n_phi, n_theta, 3(xyz))
    '''
    dg2 = winding_surface.gammadash2()
    dg1 = winding_surface.gammadash1()
    dg1xdg2 = np.cross(dg1, dg2, axis=-1)
    denom = np.sum(dg1xdg2**2, axis=-1)
    # grad phi
    grad1 = np.cross(dg2, dg1xdg2, axis=-1)/denom[:,:,None]
    # grad theta
    grad2 = np.cross(dg1, -dg1xdg2, axis=-1)/denom[:,:,None]
    return(grad1, grad2)

def norm_helper(winding_surface):
    '''
    This is a helper method that calculates the following quantities:
    - normal_vec: The normal vector N
    - normN_prime_2d: The normal vector's length, |N|
    - inv_normN_prime_2d: 1/|N|
    Shape: (n_phi, n_theta, 3(xyz)); (n_phi, n_theta); (n_phi, n_theta)
    '''
    # Length of the non-unit WS normal vector |N|,
    # its inverse (1/|N|) and its inverse's derivatives
    # w.r.t. phi(phi) and theta
    # Not to be confused with the normN (plasma surface Jacobian)
    # in Regcoil.
    normal_vec = winding_surface.normal()
    normN_prime_2d = np.sqrt(np.sum(normal_vec**2, axis=-1)) # |N|
    inv_normN_prime_2d = 1/normN_prime_2d # 1/|N|
    return(
        normal_vec,
        normN_prime_2d,
        inv_normN_prime_2d
    )

def dga_inv_n_dashb(winding_surface):
    ''' 
    This is a helper method that calculates the following quantities:
    - dg1_inv_n_dash1: d[(1/|n|)(dgamma/dphi)]/dphi
    - dg1_inv_n_dash2: d[(1/|n|)(dgamma/dphi)]/dtheta
    - dg2_inv_n_dash1: d[(1/|n|)(dgamma/dtheta)]/dphi
    - dg2_inv_n_dash2: d[(1/|n|)(dgamma/dtheta)]/dtheta
    Shape: (n_phi, n_theta, 3(xyz))
    '''
    # gammadash1() calculates partial r/partial phi. Keep in mind that the angles
    # in simsopt go from 0 to 1.
    # Shape: (n_phi, n_theta, 3(xyz))
    dg2 = winding_surface.gammadash2()
    dg1 = winding_surface.gammadash1()
    dg22 = winding_surface.gammadash2dash2()
    dg11 = winding_surface.gammadash1dash1()
    dg12 = winding_surface.gammadash1dash2()

    # Because Phi is defined around the unit normal, rather 
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
    normal_vec, _, inv_normN_prime_2d = norm_helper(winding_surface)

    # Derivatives of 1/|N|:
    # d/dx(1/sqrt(f(x)^2 + g(x)^2 + h(x)^2)) 
    # = (-f(x)f'(x) - g(x)g'(x) - h(x)h'(x))
    # /(f(x)^2 + g(x)^2 + h(x)^2)^(3/2)
    denominator = np.sum(normal_vec**2, axis=-1)**1.5
    nominator_inv_normN_prime_2d_dash1 = -np.sum(normal_vec*normaldash1, axis=-1)
    nominator_inv_normN_prime_2d_dash2 = -np.sum(normal_vec*normaldash2, axis=-1)
    inv_normN_prime_2d_dash1 = nominator_inv_normN_prime_2d_dash1/denominator
    inv_normN_prime_2d_dash2 = nominator_inv_normN_prime_2d_dash2/denominator
    
    # d[(1/|n|)(dgamma/dphi)]/dphi
    dg1_inv_n_dash1 = dg11*inv_normN_prime_2d[:,:,None] + dg1*inv_normN_prime_2d_dash1[:,:,None] 
    # d[(1/|n|)(dgamma/dphi)]/dtheta
    dg1_inv_n_dash2 = dg12*inv_normN_prime_2d[:,:,None] + dg1*inv_normN_prime_2d_dash2[:,:,None] 
    # d[(1/|n|)(dgamma/dtheta)]/dphi
    dg2_inv_n_dash1 = dg12*inv_normN_prime_2d[:,:,None] + dg2*inv_normN_prime_2d_dash1[:,:,None] 
    # d[(1/|n|)(dgamma/dtheta)]/dtheta
    dg2_inv_n_dash2 = dg22*inv_normN_prime_2d[:,:,None] + dg2*inv_normN_prime_2d_dash2[:,:,None] 
    return(
        dg1_inv_n_dash1,
        dg1_inv_n_dash2,
        dg2_inv_n_dash1,
        dg2_inv_n_dash2 
    )

def unitnormaldash(winding_surface):
    ''' 
    This is a helper method that calculates the following quantities:
    - unitnormaldash1: d unitnormal/dphi
    - unitnormaldash2: d unitnormal/dtheta
    Shape: (n_phi, n_theta, 3(xyz))
    '''
    _, _, inv_normN_prime_2 = norm_helper(winding_surface)
    (
        dg1_inv_n_dash1, dg1_inv_n_dash2,
        _, _ # dg2_inv_n_dash1, dg2_inv_n_dash2
    ) = dga_inv_n_dashb(winding_surface)

    dg2 = winding_surface.gammadash2()
    dg1_inv_n = winding_surface.gammadash1() * inv_normN_prime_2[:, :, None]
    dg22 = winding_surface.gammadash2dash2()
    dg12 = winding_surface.gammadash1dash2()
    unitnormaldash1 = (
        np.cross(dg1_inv_n_dash1, dg2, axis=-1)
        + np.cross(dg1_inv_n, dg12, axis=-1)
    )
    unitnormaldash2 = (
        np.cross(dg1_inv_n_dash2, dg2, axis=-1)
        + np.cross(dg1_inv_n, dg22, axis=-1)
    )
    return(unitnormaldash1, unitnormaldash2)

def Kdash_helper(cp:CurrentPotentialFourier, current_scale):
    '''
    Calculates the following quantity
    - Kdash1_sv_op, Kdash2_sv_op: 
    Partial derivatives of K in term of Phi (current potential) harmonics.
    Shape: (n_phi, n_theta, 3(xyz), n_dof)
    - Kdash1_const, Kdash2_const: 
    Partial derivatives of K due to secular terms (net poloidal/toroidal 
    currents). 
    Shape: (n_phi, n_theta, 3(xyz))
    - trig_m_i_n_i, trig_diff_m_i_n_i: 
    IFT operator that transforms even/odd derivatives of Phi harmonics
    produced by partial_* (see below). 
    Shape: (n_phi, n_theta, n_dof)
    - partial_theta, partial_phi, ... ,partial_theta_phi,
    A partial derivative operators that works by multiplying the harmonic
    coefficients of Phi by its harmonic number and a sign, depending whether
    the coefficient is sin or cos. DOES NOT RE-ORDER the coefficients
    into the simsopt conventions. Therefore, IFT for such derivatives 
    must be performed with trig_m_i_n_i and trig_diff_m_i_n_i (see above).
    '''
    # The uniform index for phi contains first sin Fourier 
    # coefficients, then optionally cos is stellsym=False.
    n_harmonic = len(cp.m)
    iden = np.identity(n_harmonic)
    winding_surface = cp.winding_surface
    _, normN_prime_2d, inv_normN_prime_2d = norm_helper(winding_surface)
    # Shape: (n_phi, n_theta)
    phi_grid = np.pi*2*winding_surface.quadpoints_phi[:, None]
    theta_grid = np.pi*2*winding_surface.quadpoints_theta[None, :]
    # When stellsym is enabled, Phi is a sin fourier series.
    # After a derivative, it becomes a cos fourier series.
    if winding_surface.stellsym:
        trig_choice = 1
    # Otherwise, it's a sin-cos series. After a derivative,
    # it becomes a cos-sin series.
    else:
        trig_choice = np.repeat([1,-1], n_harmonic//2)
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
    # Fourier derivatives
    partial_theta = cp.m*trig_choice*iden*2*np.pi
    partial_phi = -cp.n*trig_choice*iden*cp.nfp*2*np.pi
    partial_theta_theta = -cp.m**2*iden*(2*np.pi)**2
    partial_phi_phi = -(cp.n*cp.nfp)**2*iden*(2*np.pi)**2
    partial_theta_phi = cp.n*cp.nfp*cp.m*iden*(2*np.pi)**2
    # Some quantities
    (
        dg1_inv_n_dash1,
        dg1_inv_n_dash2,
        dg2_inv_n_dash1,
        dg2_inv_n_dash2 
    ) = dga_inv_n_dashb(winding_surface)
    # Operators that generates the derivative of K
    # Note the use of trig_diff_m_i_n_i for inverse
    # FT following odd-order derivatives.
    # Shape: (n_phi, n_theta, 3(xyz), n_dof)
    Kdash2_sv_op = (
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
    Kdash2_sv_op = np.swapaxes(Kdash2_sv_op, 2, 3)
    Kdash1_sv_op = (
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
    Kdash1_sv_op = np.swapaxes(Kdash1_sv_op, 2, 3)


    G = cp.net_poloidal_current_amperes * current_scale
    I = cp.net_toroidal_current_amperes * current_scale
    # Constant components of K's partial derivative.
    # Shape: (n_phi, n_theta, 3(xyz))
    Kdash1_const = \
        dg2_inv_n_dash1*G \
        -dg1_inv_n_dash1*I
    Kdash2_const = \
        dg2_inv_n_dash2*G \
        -dg1_inv_n_dash2*I
    return(
        Kdash1_sv_op, 
        Kdash2_sv_op, 
        Kdash1_const,
        Kdash2_const,
        trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_theta,
        partial_phi,
        partial_theta_theta,
        partial_phi_phi,
        partial_theta_phi,
    )