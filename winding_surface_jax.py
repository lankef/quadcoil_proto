
import jax.numpy as jnp
# import matplotlib.pyplot as plt
from jax import jit, lax, vmap
from functools import partial
from interpax import interp1d
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

# Remove move all the nans from an 1d array to the end of the array
def move_nans(xs):
    def f_append(carry, x):
        # Carry has the same size as xs. x is scalar.
        carry = jnp.where(jnp.tile(jnp.isnan(x), len(carry)), carry, replace_first_nan(carry, x))
        return(carry, x)

    return(lax.scan(f=f_append, init=jnp.full_like(xs, jnp.nan), xs=xs)[0])

def move_nans_and_pad(xs):
    xs_new = move_nans(xs)
    xs_new = jnp.where(jnp.isnan(xs_new), xs_new[0], xs_new)
    return(jnp.append(xs_new, xs_new[0]))

def move_nans_and_zero(xs):
    xs_new = move_nans(xs)
    xs_new = jnp.where(jnp.isnan(xs_new), 0, xs_new)
    return(xs_new)

def dof_to_rz_op(
        phi_fit, theta_fit, 
        nfp, stellsym,
        mpol:int=10, ntor:int=10):
    m_c = jnp.concatenate([
        jnp.zeros(ntor+1),
        jnp.repeat(jnp.arange(1, mpol+1), ntor*2+1)
    ])
    m_s = jnp.concatenate([
        jnp.zeros(ntor),
        jnp.repeat(jnp.arange(1, mpol+1), ntor*2+1)
    ])
    n_c = jnp.concatenate([
        jnp.arange(0, ntor+1),
        jnp.tile(jnp.arange(-ntor, ntor+1), mpol)
    ])
    n_s = jnp.concatenate([
        jnp.arange(1, ntor+1),
        jnp.tile(jnp.arange(-ntor, ntor+1), mpol)
    ])
    # 2 arrays of shape [*stack of mn, nphi, ntheta]
    # consisting of 
    # [
    #     cos(m theta - nfp * n * phi)
    #     ...
    # ]
    # [
    #     sin(m theta - nfp * n * phi)
    #     ...
    # ]
    cmn = jnp.cos(
        m_c[:, None, None] * theta_fit[None, :, :] * jnp.pi * 2
        - n_c[:, None, None] * phi_fit[None, :, :] * jnp.pi * 2 * nfp
    )
    smn = jnp.sin(
        m_s[:, None, None] * theta_fit[None, :, :] * jnp.pi * 2
        - n_s[:, None, None] * phi_fit[None, :, :] * jnp.pi * 2 * nfp
    )
    m_2_n_2 = jnp.concatenate([m_c, m_s]) ** 2 + jnp.concatenate([n_c, n_s]) ** 2
    if not stellsym:
        m_2_n_2 = jnp.tile(m_2_n_2, 2)
    # Stellsym SurfaceRZFourier's dofs consists of 
    # [rc, zs]
    # Non-stellsym SurfaceRZFourier's dofs consists of 
    # [rc, rs, zc, zs]
    # Here we construct operators that maps 
    # dofs -> r and z of known, valid, expanded quadpoints.
    if stellsym:
        r_operator = cmn
        z_operator = smn
    else:
        r_operator = jnp.concatenate([cmn, smn], axis=0)
        z_operator = jnp.concatenate([cmn, smn], axis=0)
    r_operator_padded = jnp.concatenate([r_operator, jnp.zeros_like(z_operator)], axis=0)
    z_operator_padded = jnp.concatenate([jnp.zeros_like(r_operator), z_operator], axis=0)

    # overall operator
    # has shape 
    # [ndof, nphi, ntheta, 2(r, z)]
    # maps a [ndof] array
    # to a [nphi, ntheta, 2(r, z)] array.
    A_lstsq = jnp.concatenate([r_operator_padded[:, :, :, None], z_operator_padded[:, :, :, None]], axis=3)
    return(A_lstsq, m_2_n_2)

@partial(jit, static_argnames=['mpol', 'ntor', 'lam_tikhnov', 'lam_gaussian', 'stellsym', 'nfp'])
def fit_surfacerzfourier(
        phi_fit, theta_fit, 
        r_fit, z_fit, 
        nfp:int, stellsym:bool, 
        mpol:int=10, ntor:int=10, 
        lam_tikhnov=0.8, lam_gaussian=0.8,
        custom_weight=1,):

    A_lstsq, m_2_n_2 = dof_to_rz_op(
        theta_fit=theta_fit, 
        phi_fit=phi_fit,
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
    print('weight', weight.shape)
    A_lstsq = A_lstsq * weight[None, :, :, None]
    b_lstsq = b_lstsq * weight[:, :, None]
    A_lstsq = A_lstsq.reshape(A_lstsq.shape[0], -1).T
    b_lstsq = b_lstsq.flatten()

    # Tikhnov regularization for higher harmonics
    lam = lam_tikhnov * jnp.average(A_lstsq.T.dot(b_lstsq)) * jnp.diag(m_2_n_2)
    dofs_expand, resid, rank, s = jnp.linalg.lstsq(A_lstsq.T.dot(A_lstsq) + lam, A_lstsq.T.dot(b_lstsq))
    return(dofs_expand, resid)


# An approximation for unit normal.
# and include the endpoints
gen_rot_matrix = lambda theta: jnp.array([
    [jnp.cos(theta), -jnp.sin(theta), 0],
    [jnp.sin(theta),  jnp.cos(theta), 0],
    [0,              0,             1]
])

# For 2d inputs
move_nans_and_pad_vmap = vmap(move_nans_and_pad)

@partial(jit, static_argnames=[
    'nfp', 'stellsym', 
    'n_theta', 'n_phi', 'n_theta_interp', 
    'mpol', 'ntor', 
    'tol_expand', 'interp1d_method', 'lam_tikhnov', 'simple_mode'
])
def gen_winding_surface(
        gamma_plasma, d_expand, 
        nfp, stellsym,
        unitnormal=None,
        n_theta=64, n_phi=64,
        n_theta_interp=64, 
        mpol=10, ntor=10,
        tol_expand=0.9,
        interp1d_method='cubic',
        lam_tikhnov=0.9,
        simple_mode=False
        ):

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
        next_fp = gamma_plasma[:n_phi//2] @ rotation_matrix.T
        last_fp = gamma_plasma[n_phi//2:] @ rotation_matrix_neg.T
    else:
        next_fp = gamma_plasma @ rotation_matrix.T
        last_fp = gamma_plasma @ rotation_matrix_neg.T
        # Copy the gamma from the next and last fp.
    gamma_plasma_dist = jnp.concatenate([last_fp, gamma_plasma, next_fp], axis=0)

    r_expand = jnp.sqrt(gamma_plasma_expand[:, :, 1]**2 + gamma_plasma_expand[:, :, 0]**2)
    phi_expand = jnp.arctan2(gamma_plasma_expand[:, :, 1], gamma_plasma_expand[:, :, 0]) / jnp.pi / 2 
    z_expand = gamma_plasma_expand[:, :, 2]

    stage_1_dofs, stage_1_resid = fit_surfacerzfourier(
        mpol=mpol,
        ntor=ntor,
        phi_fit=phi_expand,
        theta_fit=jnp.linspace(0, 1, phi_expand.shape[1])[None, :]+jnp.zeros_like(phi_expand),
        r_fit=r_expand,
        z_fit=z_expand,
        nfp=nfp,
        stellsym=stellsym, 
        lam_tikhnov=0., lam_gaussian=0.,
    )


    quadpoints_phi_stage_1=jnp.linspace(0, 1, n_phi, endpoint=False)/nfp
    quadpoints_theta_stage_1=jnp.linspace(0, 1, n_theta, endpoint=False)

    A_eval, _ = dof_to_rz_op(
        phi_fit=quadpoints_phi_stage_1[:, None] + jnp.zeros((1, len(quadpoints_theta_stage_1))),
        theta_fit=quadpoints_theta_stage_1[None, :] + jnp.zeros((len(quadpoints_phi_stage_1), 1)),
        nfp=nfp, stellsym=stellsym,
        mpol=mpol, ntor=ntor,
        )

    rz_stage_1 = jnp.moveaxis(A_eval, 0, -1)@stage_1_dofs
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
    
    # For debugging
    # gamma_remove_invalid = jnp.where(
    #     min_dist_stage_1[:, :, None] + jnp.zeros_like(gamma_stage_1) < tol_expand * d_expand, 
    #     jnp.nan, 
    #     gamma_stage_1
    # )

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


    ''' DEBUGGING '''
    # x_interp = jnp.cos(quadpoints_phi_stage_1[:, None] * jnp.pi * 2) * r_interp
    # y_interp = jnp.sin(quadpoints_phi_stage_1[:, None] * jnp.pi * 2) * r_interp
    # gamma_interp = jnp.concatenate([
    #     x_interp[:, :-1, None],
    #     y_interp[:, :-1, None],
    #     z_interp[:, :-1, None],
    # ], axis=-1)
    # print(gamma_stage_1.shape)
    # print(gamma_interp.shape)
    # gamma_and_field_to_vtk(gamma_plasma, unitnormal, 'gamma_unit_normal.vtk')
    # gamma_and_field_to_vtk(gamma_plasma, unitnormal * d_expand, 'gamma_expand_vector.vtk')
    # gamma_and_scalar_field_to_vtk(gamma_plasma_expand, phi_expand, 'gamma_expand_phi.vtk')
    # gamma_to_vtk(gamma_plasma_expand, 'gamma_expand.vtk')
    # gamma_to_vtk(gamma_plasma, 'gamma_plasma.vtk')
    # gamma_to_vtk(gamma_stage_1, 'gamma_stage_1.vtk')
    # gamma_to_vtk(gamma_interp, 'gamma_interp.vtk')

    dofs_expand, resid = fit_surfacerzfourier(
        mpol=mpol,
        ntor=ntor,
        theta_fit=arclen_theta_interp_norm, # theta_interp
        phi_fit=quadpoints_phi_stage_1[:, None] + jnp.zeros_like(r_interp),
        r_fit=r_interp,
        z_fit=z_interp,
        nfp=nfp, stellsym=stellsym,
        lam_tikhnov=lam_tikhnov, lam_gaussian=0.,
    )

    return(dofs_expand)