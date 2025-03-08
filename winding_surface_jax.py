
import jax.numpy as jnp
# import matplotlib.pyplot as plt
from jax import jit, lax, vmap
from jax.lax import while_loop
from functools import partial
from interpax import interp1d
import lineax as lx
import numpy as np
import pyvista as pv

from surfacerzfourier_jax import dof_to_rz_op, dof_to_gamma

''' Plotting '''
# import pyvista as pv
# import numpy as np
def gamma_to_vtk(data, name):
    # Assuming 'data' is your array of shape (N, M, 3) with [:,:,0] = r, [:,:,1] = phi, and [:,:,2] = z
    x = data[:, :, 0]
    y = data[:, :, 1]
    z = data[:, :, 2]

    # Flatten the arrays for grid points
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]

    # Create the structured grid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (x.shape[1], x.shape[0], 1)  # Set the grid dimensions, with 1 as the third dimension

    # Save as a VTK file
    grid.save(name)

def gamma_and_field_to_vtk(gamma, f, name):
    # Assuming gamma and f are numpy arrays of shape (m, n, 3)
    m, n, _ = gamma.shape

    # Reshape gamma to a flat list of points
    points = gamma.reshape(-1, 3)

    # Reshape f to a flat list of vectors
    vectors = f.reshape(-1, 3)

    # Create a structured grid in PyVista
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [m, n, 1]  # Structured grid dimensions
    grid["f"] = vectors  # Add the vector field to the grid

    # Save the grid to a VTK file for ParaView
    grid.save(name)

def gamma_and_scalar_field_to_vtk(xyz_data, scalar_data, name):
    # Assuming `xyz_data` is your array of shape (N, M, 3) storing the x, y, z coordinates
    # and `scalar_data` is your array of shape (N, M) storing the scalar field
    x = xyz_data[:, :, 0]
    y = xyz_data[:, :, 1]
    z = xyz_data[:, :, 2]

    # Flatten the coordinate arrays and the scalar field for structured grid points
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]
    scalars = scalar_data.ravel()

    # Create the structured grid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (x.shape[1], x.shape[0], 1)  # Set dimensions with 1 in the third axis

    # Add scalar field as point data
    grid["Scalar Field"] = scalars

    # Save as a VTK file
    grid.save(name)

''' Surface fitting '''

# Removes the first nan in an 1d array
def replace_first_nan(xs, val):
    def f_modify_first_nan(carry, x):
        # Carry contains 2 elements: element to apply and a state var.
        content = carry[0]
        state = carry[1]
        carry = jnp.where(jnp.logical_and(jnp.isnan(x), state==1), jnp.array([content, 0]), carry)
        x = jnp.where(jnp.logical_and(jnp.isnan(x), state==1), content, x)
        return(carry, x)
    return(lax.scan(f=f_modify_first_nan, init=jnp.array([val, 1]), xs=xs)[1])

# Move all the nans from an 1d array to the end of the array
def move_nans(xs):
    def f_append(carry, x):
        # Carry has the same size as xs. x is scalar.
        carry = jnp.where(jnp.tile(jnp.isnan(x), len(carry)), carry, replace_first_nan(carry, x))
        return(carry, x)

    return(lax.scan(f=f_append, init=jnp.full_like(xs, jnp.nan), xs=xs)[0])

# replace all nans with the first element
def move_nans_and_pad(xs):
    xs_new = move_nans(xs)
    xs_new = jnp.where(jnp.isnan(xs_new), xs_new[0], xs_new)
    return(jnp.append(xs_new, xs_new[0]))

def move_nans_and_zero(xs):
    xs_new = move_nans(xs)
    xs_new = jnp.where(jnp.isnan(xs_new), 0, xs_new)
    return(xs_new)

def winding_surface_field_Bn(points_plasma, points_coil, normal_plasma, normal_coil, stellsym, zeta_coil, theta_coil, ndofs, m, n, nfp):
    # Ensure inputs are NumPy arrays
    points_plasma = jnp.asarray(points_plasma)
    points_coil = jnp.asarray(points_coil)
    normal_plasma = jnp.asarray(normal_plasma)
    normal_coil = jnp.asarray(normal_coil)
    zeta_coil = jnp.asarray(zeta_coil)
    theta_coil = jnp.asarray(theta_coil)
    m = jnp.asarray(m)
    n = jnp.asarray(n)

    # Precompute constants
    fak = 1e-7  # mu0 / (4 * pi)

    # Calculate gij
    diff = points_plasma[:, None, :] - points_coil[None, :, :]
    rmag2 = jnp.sum(diff**2, axis=-1)
    rmag_inv = 1.0 / jnp.sqrt(rmag2)
    rmag_inv_3 = rmag_inv**3
    rmag_inv_5 = rmag_inv**5

    npdotnc = jnp.sum(normal_plasma[:, None, :] * normal_coil[None, :, :], axis=-1)
    rdotnp = jnp.sum(diff * normal_plasma[:, None, :], axis=-1)
    rdotnc = jnp.sum(diff * normal_coil[None, :, :], axis=-1)

    gij = fak * (npdotnc * rmag_inv_3 - 3.0 * rdotnp * rdotnc * rmag_inv_5)

    # Calculate gj
    angle = 2 * jnp.pi * m[:, None] * theta_coil - 2 * jnp.pi * n[:, None] * zeta_coil * nfp
    sphi = jnp.sin(angle)  # Shape: (len(m), num_coil)
    cphi = jnp.cos(angle)  # Shape: (len(m), num_coil)

    # Reshape gij for compatibility: (num_plasma, 1, num_coil)
    gij_expanded = gij[:, None, :]  # Add an axis for compatibility with sphi and cphi

    # Compute gj using broadcasting and summing over coils
    gj_sin = jnp.sum(gij_expanded * sphi[None, :, :], axis=-1)  # Shape: (num_plasma, len(m))
    gj = gj_sin

    if not stellsym:
        gj_cos = jnp.sum(gij_expanded * cphi[None, :, :], axis=-1)  # Shape: (num_plasma, len(m))
        gj = jnp.concatenate([gj, gj_cos], axis=1)  # Shape: (num_plasma, 2 * len(m))

    # Calculate Ajk
    normal_norms = jnp.linalg.norm(normal_plasma, axis=-1, keepdims=True)  # Shape: (num_plasma, 1)
    gj_normalized = gj / normal_norms  # Normalize gj by normal_plasma

    Ajk = jnp.dot(gj_normalized.T, gj_normalized)  # Shape: (ndofs, ndofs)

    return gj, Ajk

def winding_surface_field_Bn_GI(points_plasma, points_coil, normal_plasma, zeta_coil, theta_coil, G, I, gammadash1_coil, gammadash2_coil):
    # Ensure inputs are JAX arrays
    points_plasma = jnp.asarray(points_plasma)
    points_coil = jnp.asarray(points_coil)
    normal_plasma = jnp.asarray(normal_plasma)
    gammadash1_coil = jnp.asarray(gammadash1_coil)
    gammadash2_coil = jnp.asarray(gammadash2_coil)

    # Constants
    fak = 1e-7  # mu0 / (8 * pi^2)

    # Normalize normal_plasma vectors
    nmag = jnp.linalg.norm(normal_plasma, axis=-1, keepdims=True)
    normal_plasma_normalized = normal_plasma / nmag

    # Vectorized computation of rx, ry, rz (broadcasting over plasma and coil points)
    diff = points_plasma[:, None, :] - points_coil[None, :, :]  # Shape: (num_plasma, num_coil, 3)

    # Compute rmag_inv and rmag_inv_3
    rmag = jnp.linalg.norm(diff, axis=-1)  # Shape: (num_plasma, num_coil)
    rmag_inv = 1.0/rmag  # Shape: (num_plasma, num_coil)
    rmag_inv_3 = rmag_inv**3  # Shape: (num_plasma, num_coil)

    # Compute GI vector
    GI = G * gammadash2_coil - I * gammadash1_coil  # Shape: (num_coil, 3)

    # Compute GI cross r
    GI_cross_r = jnp.cross(GI[None, :,  :], diff, axis=-1)  # Shape: (num_plasma, num_coil, 3)

    # Dot product of GI_cross_r with normal_plasma
    GIcrossr_dotn = jnp.sum(GI_cross_r * normal_plasma_normalized[:, None, :], axis=-1)  # Shape: (num_plasma, num_coil)

    # Compute B_GI
    B_GI = jnp.sum(fak * GIcrossr_dotn * rmag_inv_3, axis=1)  # Shape: (num_plasma,)

    return B_GI

def cp_make_mn(mpol, ntor, stellsym):
    """
    Make the list of m and n values. Equivalent to CurrentPotential._make_mn.
    """
    m1d = jnp.arange(mpol + 1)
    n1d = jnp.arange(-ntor, ntor + 1)
    n2d, m2d = jnp.meshgrid(n1d, m1d)
    m0 = m2d.flatten()[ntor:]
    n0 = n2d.flatten()[ntor:]
    m = m0[1::]
    n = n0[1::]

    if not stellsym:
        m = jnp.append(m, m)
        n = jnp.append(n, n)
    return(m, n)

# def dof_to_normal_op(
#     phi_grid, theta_grid, 
#     nfp, stellsym,
#     mpol:int=10, ntor:int=10):
#     A_gammadash1 = dof_to_gamma_op(
#         phi_grid=phi_grid, 
#         theta_grid=theta_grid, 
#         nfp=nfp, 
#         stellsym=stellsym,
#         dash1_order=1, 
#         mpol=mpol, 
#         ntor=ntor
#     )
#     A_gammadash2 = dof_to_gamma_op(
#         phi_grid=phi_grid, 
#         theta_grid=theta_grid, 
#         nfp=nfp, 
#         stellsym=stellsym,
#         dash2_order=1, 
#         mpol=mpol, 
#         ntor=ntor
#     )
#     return(jnp.cross(A_gammadash1[:, :, :, :, None], A_gammadash2[:, :, :, None, :], axis=-3))

@partial(jit, static_argnames=['nfp', 'stellsym', 'mpol', 'ntor', 'lam_tikhnov', 'lam_gaussian',])
def fit_surfacerzfourier(
        phi_grid, theta_grid, 
        r_fit, z_fit, 
        nfp:int, stellsym:bool, 
        mpol:int=10, ntor:int=10, 
        lam_tikhnov=0.8, lam_gaussian=0.8,
        custom_weight=1,):

    A_lstsq, m_2_n_2 = dof_to_rz_op(
        theta_grid=theta_grid, 
        phi_grid=phi_grid,
        nfp=nfp, 
        stellsym=stellsym, 
        mpol=mpol, 
        ntor=ntor
    )
    
    b_lstsq = jnp.concatenate([r_fit[:, :, None], z_fit[:, :, None]], axis=2)
    # Weight each point by the sum of the length of the two 
    # poloidal segments that each vertex is attached to
    # weight = jacobian # 
    max_r_slice = jnp.max(r_fit, axis=1)[:, None]
    min_r_slice = jnp.min(r_fit, axis=1)[:, None]
    weight = jnp.exp(-(lam_gaussian * (r_fit-min_r_slice)/(max_r_slice-min_r_slice))**2) * custom_weight
    # A and b of the lstsq problem.
    # A_lstsq is a function of phi_grid and theta_grid
    # b_lstsq is differentiable.
    # A_lstsq has shape: [nphi, ntheta, 2(rz), ndof]
    # b_lstsq has shape: [nphi, ntheta, 2(rz)]
    A_lstsq = A_lstsq * weight[:, :, None, None]
    b_lstsq = b_lstsq * weight[:, :, None]
    A_lstsq = A_lstsq.reshape(-1, A_lstsq.shape[-1])
    b_lstsq = b_lstsq.flatten()

    # Tikhnov regularization for higher harmonics
    lam = lam_tikhnov * jnp.average(A_lstsq.T.dot(b_lstsq)) * jnp.diag(m_2_n_2)
    
    # The lineax call fulfills the same purpose as the following:
    # dofs_expand, resid, rank, s = jnp.linalg.lstsq(A_lstsq.T.dot(A_lstsq) + lam, A_lstsq.T.dot(b_lstsq))
    # but is faster and more robust to gradients.
    operator = lx.MatrixLinearOperator(A_lstsq.T.dot(A_lstsq) + lam)
    solver = lx.QR()  # or lx.AutoLinearSolver(well_posed=None)
    solution = lx.linear_solve(operator, A_lstsq.T.dot(b_lstsq), solver)
    return(solution.value)

# An approximation for unit normal.
# and include the endpoints
gen_rot_matrix = lambda theta: jnp.array([
    [jnp.cos(theta), -jnp.sin(theta), 0],
    [jnp.sin(theta),  jnp.cos(theta), 0],
    [0,              0,             1]
])

# For 2d inputs
move_nans_and_pad_vmap = vmap(move_nans_and_pad)

def remove_dup(arr):
    # Remove duplicate items one at a time
    def iter(arr):
        arr_p1 = jnp.roll(arr, 1)
        arr_m1 = jnp.roll(arr, -1)
        return(jnp.where(arr_m1 == arr, (arr_p1 + arr) / 2, arr))

    # When true, keep looping
    def cond(arr):
        arr_m1 = jnp.roll(arr, -1)
        return(jnp.any(arr_m1 == arr))

    return(while_loop(cond, iter, arr))

remove_dup_vmap = vmap(remove_dup)

# default values
tol_expand_default = 0.9
lam_tikhnov_default = 0.2

@partial(jit, static_argnames=[
    'nfp', 'stellsym', 
    'mpol', 'ntor', 
    'tol_expand', 'lam_tikhnov'
])
def gen_winding_surface_atan(
        gamma_plasma, d_expand, 
        nfp, stellsym,
        unitnormal=None,
        mpol=10, ntor=10,
        tol_expand=tol_expand_default,
        lam_tikhnov=lam_tikhnov_default,
        ):
    # A simple winding surface generator with less intermediate quantities.
    # only works for large offset distances, where center (from the unweighted
    # avg of the quadrature points' rz coordinate) of the offset surface's rz cross sections
    # lay within the cross sections. 

    theta = 2 * jnp.pi / nfp
    rotation_matrix = gen_rot_matrix(theta)
    rotation_matrix_neg = gen_rot_matrix(-theta)

    # Approximately calculating the normal vector. Alternatively, the normal
    # can be provided, but this will make the Jacobian matrix larger and lead to longer compile time.
    if unitnormal is None:
        xyz_rotated = gamma_plasma[0, :, :] @ rotation_matrix.T
        gamma_plasma_phi_rolled = jnp.append(gamma_plasma[1:, :, :], xyz_rotated[None, :, :], axis=0)
        delta_phi = gamma_plasma_phi_rolled - gamma_plasma
        delta_theta = jnp.roll(gamma_plasma, 1, axis=1) - gamma_plasma
        normal_approx = jnp.cross(delta_theta, delta_phi)
        unitnormal = normal_approx / jnp.linalg.norm(normal_approx, axis=-1)[:,:,None]
    
    # Copy the next field period 
    if stellsym:
        # If stellsym, then only use half of the field period for surface fitting
        len_phi = gamma_plasma.shape[0]//2
        gamma_plasma_expand = (
            gamma_plasma[:len_phi] 
            + unitnormal[:len_phi] * d_expand)
        next_fp = gamma_plasma[:len_phi] @ rotation_matrix.T
        last_fp = gamma_plasma[len_phi:] @ rotation_matrix_neg.T
        gamma_plasma_dist = jnp.concatenate([last_fp, gamma_plasma], axis=0)
        # finding center to generate poloidal parameterization
        r_plasma = jnp.sqrt(gamma_plasma[:len_phi, :, 1]**2 + gamma_plasma[:len_phi, :, 0]**2)
        z_plasma = gamma_plasma[:len_phi, :, 2]
        r_center = jnp.average(r_plasma, axis=-1)
        z_center = jnp.average(z_plasma, axis=-1)
    else:
        gamma_plasma_expand = gamma_plasma + unitnormal * d_expand
        next_fp = gamma_plasma @ rotation_matrix.T
        last_fp = gamma_plasma @ rotation_matrix_neg.T
        # Copy the gamma from the next and last fp.
        gamma_plasma_dist = jnp.concatenate([last_fp, gamma_plasma, next_fp], axis=0)
        # finding center to generate poloidal parameterization
        r_plasma = jnp.sqrt(gamma_plasma[:, :, 1]**2 + gamma_plasma[:, :, 0]**2)
        z_plasma = gamma_plasma[:, :, 2]
        r_center = jnp.average(r_plasma, axis=-1)
        z_center = jnp.average(z_plasma, axis=-1)
    # The original uniform offset. Has self-intersections.
    # Tested to be differentiable.
    r_expand = jnp.sqrt(gamma_plasma_expand[:, :, 1]**2 + gamma_plasma_expand[:, :, 0]**2)
    phi_expand = jnp.arctan2(gamma_plasma_expand[:, :, 1], gamma_plasma_expand[:, :, 0]) / jnp.pi / 2 
    z_expand = gamma_plasma_expand[:, :, 2]


    min_dist_expand = jnp.min((
        jnp.linalg.norm(gamma_plasma_expand[:, :, None, None, :] - gamma_plasma_dist[None, None, :, :, :], axis=-1)
    ), axis=(2, 3))

    weight_remove_invalid = jnp.where(
        min_dist_expand[:, :] < tol_expand * d_expand, 
        0., 
        1.
    )
    # print('gamma_plasma_expand', gamma_plasma_expand.shape)
    # print('gamma_plasma_dist', gamma_plasma_dist.shape)
    # print('z_expand', z_expand.shape)
    # print('z_center', z_center.shape)
    # print('r_expand', r_expand.shape)
    # print('r_center', r_center.shape)
    theta_atan = jnp.arctan2(z_expand-z_center[:, None], r_expand-r_center[:, None])/jnp.pi/2
    # gamma_and_scalar_field_to_vtk(weight_remove_invalid[:, :, None] * gamma_plasma_expand, theta_atan, 'ws_new_to_fit.vts')
    dofs_expand = fit_surfacerzfourier(
        mpol=mpol,
        ntor=ntor,
        theta_grid=theta_atan, # theta_interp
        phi_grid=phi_expand,
        r_fit=r_expand,
        z_fit=z_expand,
        nfp=nfp, stellsym=stellsym,
        lam_tikhnov=lam_tikhnov, lam_gaussian=0.,
        custom_weight=weight_remove_invalid,
    )

    return(dofs_expand)


# This is necessary to calculate gammadashes.

# @partial(jit, static_argnames=[
#     'nfp', 
#     'stellsym', 
#     'mpol_plasma',
#     'ntor_plasma',
#     'mpol_winding',
#     'ntor_winding',
# ])
def plasma_dofs_to_winding_dofs(
    # Dofs
    plasma_dofs,
    # Equilibrium and related parameters
    # Coil parameters
    coil_plasma_distance,
    # Numerical parameters
    nfp, 
    stellsym, 
    mpol_plasma,
    ntor_plasma,
    quadpoints_phi_plasma,
    quadpoints_theta_plasma,
    mpol_winding=10, 
    ntor_winding=10,
):

    theta_mesh_plasma, phi_mesh_plasma = jnp.meshgrid(quadpoints_theta_plasma, quadpoints_phi_plasma)

    ''' Plasma surface calculations'''
    # Quadrature points
    gamma_plasma = dof_to_gamma(
        dofs=plasma_dofs,
        phi_grid=phi_mesh_plasma, 
        theta_grid=theta_mesh_plasma, 
        nfp=nfp, 
        stellsym=stellsym, 
        mpol=mpol_plasma, ntor=ntor_plasma)
    ''' Generating winding surface '''

    winding_dofs = gen_winding_surface_atan(
        gamma_plasma=gamma_plasma, 
        d_expand=coil_plasma_distance, 
        nfp=nfp, stellsym=stellsym,
        unitnormal=None,
        mpol=mpol_winding, ntor=ntor_winding,
        tol_expand=0.9,
        lam_tikhnov=0.9,
    )
    return(winding_dofs, gamma_plasma)


@partial(jit, static_argnames=[
    'nfp', 'stellsym', 
    'n_theta', 'n_phi', 'n_theta_interp', 
    'mpol', 'ntor', 
    'tol_expand', 'interp1d_method', 'lam_tikhnov'
])
def gen_winding_surface_arclen(
        gamma_plasma, d_expand, 
        nfp, stellsym,
        unitnormal=None,
        n_theta=64, n_phi=64,
        n_theta_interp=64, 
        mpol=10, ntor=10,
        tol_expand=0.9,
        interp1d_method='cubic',
        lam_tikhnov=0.9,
        ):
    # A more complex winding surface generator robust to 
    # large plasma indent and small coil-plasma distance. 
    # DOES NOT work well with autodiff.
    # Create the rotation matrix for rotation around the z-axis. 
    # To correctly calculate coil-plasma distance, one must use at least 3 field periods.

    theta = 2 * jnp.pi / nfp
    rotation_matrix = gen_rot_matrix(theta)
    rotation_matrix_neg = gen_rot_matrix(-theta)

    # Approximately calculating the normal vector. Alternatively, the normal
    # can be provided, but this will make the Jacobian matrix larger and lead to longer compile time.
    if unitnormal is None:
        xyz_rotated = gamma_plasma[0, :, :] @ rotation_matrix.T
        gamma_plasma_phi_rolled = jnp.append(gamma_plasma[1:, :, :], xyz_rotated[None, :, :], axis=0)
        delta_phi = gamma_plasma_phi_rolled - gamma_plasma
        delta_theta = jnp.roll(gamma_plasma, 1, axis=1) - gamma_plasma
        normal_approx = jnp.cross(delta_theta, delta_phi)
        unitnormal = normal_approx / jnp.linalg.norm(normal_approx, axis=-1)[:,:,None]
    gamma_plasma_expand = gamma_plasma + unitnormal * d_expand

    
    if stellsym:
        next_fp = gamma_plasma[:gamma_plasma.shape[0]//2] @ rotation_matrix.T
        last_fp = gamma_plasma[gamma_plasma.shape[0]//2:] @ rotation_matrix_neg.T
    else:
        next_fp = gamma_plasma @ rotation_matrix.T
        last_fp = gamma_plasma @ rotation_matrix_neg.T
        # Copy the gamma from the next and last fp.
    gamma_plasma_dist = jnp.concatenate([last_fp, gamma_plasma, next_fp], axis=0)
    print('gamma_plasma_expand', gamma_plasma_expand)
    r_expand = jnp.sqrt(gamma_plasma_expand[:, :, 1]**2 + gamma_plasma_expand[:, :, 0]**2)
    phi_expand = jnp.arctan2(gamma_plasma_expand[:, :, 1], gamma_plasma_expand[:, :, 0]) / jnp.pi / 2 
    z_expand = gamma_plasma_expand[:, :, 2]
    print('phi_expand', phi_expand)

    stage_1_dofs = fit_surfacerzfourier(
        mpol=mpol,
        ntor=ntor,
        phi_grid=phi_expand,
        theta_grid=jnp.linspace(0, 1, phi_expand.shape[1])[None, :]+jnp.zeros_like(phi_expand),
        r_fit=r_expand,
        z_fit=z_expand,
        nfp=nfp,
        stellsym=stellsym, 
        lam_tikhnov=0., lam_gaussian=0.,
    )


    quadpoints_phi_stage_1=jnp.linspace(0, 1, n_phi, endpoint=False)/nfp
    quadpoints_theta_stage_1=jnp.linspace(0, 1, n_theta, endpoint=False)

    A_eval, _ = dof_to_rz_op(
        phi_grid=quadpoints_phi_stage_1[:, None] + jnp.zeros((1, len(quadpoints_theta_stage_1))),
        theta_grid=quadpoints_theta_stage_1[None, :] + jnp.zeros((len(quadpoints_phi_stage_1), 1)),
        nfp=nfp, stellsym=stellsym,
        mpol=mpol, ntor=ntor,
        )

    rz_stage_1 = A_eval@stage_1_dofs
    r_stage_1 = rz_stage_1[:, :, 0]
    z_stage_1 = rz_stage_1[:, :, 1]
    x_stage_1 = jnp.cos(quadpoints_phi_stage_1[:, None] * jnp.pi * 2) * r_stage_1
    y_stage_1 = jnp.sin(quadpoints_phi_stage_1[:, None] * jnp.pi * 2) * r_stage_1
    gamma_stage_1 = jnp.concatenate([
        x_stage_1[:, :, None], 
        y_stage_1[:, :, None], 
        z_stage_1[:, :, None]
    ], axis=-1)
    min_dist_stage_1 = jnp.min((
        jnp.linalg.norm(gamma_stage_1[:, :, None, None, :] - gamma_plasma_dist[None, None, :, :, :], axis=-1)
    ), axis=(2, 3))

    r_remove_invalid = jnp.where(
        min_dist_stage_1[:, :] < tol_expand * d_expand, 
        jnp.nan, 
        r_stage_1
    )
    z_remove_invalid = jnp.where(
        min_dist_stage_1[:, :] < tol_expand * d_expand, 
        jnp.nan, 
        z_stage_1
    )

    # First remove all nans into the end, 
    # then replace all nans with the first element
    # last append the first element to the end of the array.
    r_for_interp = move_nans_and_pad_vmap(r_remove_invalid)
    z_for_interp = move_nans_and_pad_vmap(z_remove_invalid)
    # The length of each segment is non-differentiable when zero-length segments
    # exist. This removes zero-length segments.
    
    r_for_interp = move_nans_and_pad_vmap(r_remove_invalid)
    z_for_interp = move_nans_and_pad_vmap(z_remove_invalid)

    # The recover a new set of quadrature points by interpolating 
    # from the remaining quadrature points.
    seglen_for_interp = jnp.sqrt(
        (r_for_interp[:, 1:] - r_for_interp[:, :-1])**2
        + (z_for_interp[:, 1:] - z_for_interp[:, :-1])**2
    )
    arclen_for_interp = jnp.cumsum(seglen_for_interp, axis=1)
    t_for_interp = arclen_for_interp/(arclen_for_interp[:, -1][:, None])
    t_for_interp = jnp.concatenate((jnp.zeros((t_for_interp.shape[0],1)), t_for_interp), axis=1)

    t_cubic = jnp.linspace(0, 1, n_theta_interp)[None, :] + jnp.zeros(n_phi)[:, None]

    interp_f = lambda xq, x, f: interp1d(xq, x, f, period=1, method=interp1d_method) # monotonic, catmull-rom
    interp_f_vec = vmap(interp_f, in_axes=0, out_axes=0)

    r_interp = interp_f_vec(t_cubic, t_for_interp, r_for_interp)
    z_interp = interp_f_vec(t_cubic, t_for_interp, z_for_interp)

    seglen_theta_interp = jnp.sqrt(
        (r_interp[:, 1:] - r_interp[:, :-1])**2
        + (z_interp[:, 1:] - z_interp[:, :-1])**2
    )
    arclen_theta_interp = jnp.cumsum(seglen_theta_interp, axis=1)
    arclen_theta_interp_norm = arclen_theta_interp/(arclen_theta_interp[:, -1][:, None])
    arclen_theta_interp_norm = jnp.concatenate([
        jnp.zeros((arclen_theta_interp_norm.shape[0], 1)),
        arclen_theta_interp_norm
    ], axis=1)

    # Fitting the final Fourier surface
    dofs_expand = fit_surfacerzfourier(
        mpol=mpol,
        ntor=ntor,
        theta_grid=arclen_theta_interp_norm, # theta_interp
        phi_grid=quadpoints_phi_stage_1[:, None] + jnp.zeros_like(r_interp),
        r_fit=r_interp,
        z_fit=z_interp,
        nfp=nfp, stellsym=stellsym,
        lam_tikhnov=lam_tikhnov, lam_gaussian=0.,
    )

    return(dofs_expand)


def cp_ndof_from_mpol_ntor(mpol, ntor, stellsym):
    ndof = 2 * mpol * ntor + mpol + ntor
    if not stellsym:
        ndof *= 2
    return ndof