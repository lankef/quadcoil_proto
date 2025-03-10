# Import packages.
import jax.numpy as jnp
try:
    from shapely.geometry import LineString, MultiPolygon
    from shapely.ops import unary_union, polygonize
except ImportError:
    LineString=None
    unary_union=None
    polygonize=None

# from simsopt.objectives import SquaredFlux
# from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, plot
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
# from simsoptpp import WindingSurfaceBn_REGCOIL
def avg_order_of_magnitude(x):
    # Estimating the maximum magnitude for normalizing. 
    # the appended 1 prevents this from going to zero.
    return(jnp.max(jnp.append(jnp.abs(x).flatten(), 1.)))
    # The below implementation is not robust in autdiff.
    # Remove zeros
    # x_nonzero = jnp.where(x==0, jnp.nan, x)
    # return(jnp.exp(jnp.nanmean(jnp.log(jnp.abs(x_nonzero)))))
# A helper method. When mode=0, calculates sin(x).
# Otherwise calculates cos(x)
sin_or_cos = lambda x, mode: jnp.where(mode==1, jnp.sin(x), jnp.cos(x))

''' Winding surface '''
# @SimsoptRequires(LineString is not None, "gen_union_winding_surface requires shapely package")
def callable_RZ_union(vertices_R, vertices_Z):
    vertices_R_wrapped = jnp.pad(vertices_R, (0, 1), mode='wrap')
    vertices_Z_wrapped = jnp.pad(vertices_Z, (0, 1), mode='wrap')
    vertices_RZ_i = jnp.stack((vertices_R_wrapped, vertices_Z_wrapped)).T
    linestring = LineString(vertices_RZ_i)
    polygon = unary_union(polygonize(unary_union(linestring)))
    # This union may contain multiple polygons. 
    # When this is the case, we keep only the largest.
    if type(polygon) is MultiPolygon:
        area_list = []
        poly_list = list(polygon.geoms)
        for p in poly_list:
            area_list.append(p.area)
        polygon = poly_list[jnp.argmax(area_list)]
    union_vertices = jnp.array(polygon.exterior.coords).T
    union_vertices_R_periodic = union_vertices[0]
    union_vertices_Z_periodic = union_vertices[1]
    return(
        union_vertices_R_periodic,
        union_vertices_Z_periodic,
    )

def callable_RZ_conv(vertices_R, vertices_Z):
    ConvexHull_i = ConvexHull(
        jnp.array([
            vertices_R,
            vertices_Z
        ]).T
    )
    vertices_i = ConvexHull_i.vertices
    # Obtaining vertices
    vertices_R_new = vertices_R[vertices_i]
    vertices_Z_new = vertices_Z[vertices_i]
    vertices_R_periodic = jnp.append(vertices_R_new, vertices_R_new[0])
    vertices_Z_periodic = jnp.append(vertices_Z_new, vertices_Z_new[0])
    return(vertices_R_periodic, vertices_Z_periodic)

def gen_conv_winding_surface(
        plasma_surface, 
        d_expand,
        mpol=None,
        ntor=None,
        n_phi=None,
        n_theta=None,
    ):
    return(gen_callable_winding_surface(
        callable_RZ_conv,
        plasma_surface, 
        d_expand,
        mpol,
        ntor,
        n_phi,
        n_theta,
    ))

def gen_union_winding_surface(
        plasma_surface, 
        d_expand,
        mpol=None,
        ntor=None,
        n_phi=None,
        n_theta=None,
    ):
    return(gen_callable_winding_surface(
        callable_RZ_union,
        plasma_surface, 
        d_expand,
        mpol,
        ntor,
        n_phi,
        n_theta,
    ))


# lambda_RZ must converts two arrays containing R and Z (without repeating endpoints)
# into two arrays containing R and Z WITH REPEATING ENDPOINTS
def gen_callable_winding_surface(
        callable_process_and_wrap_RZ,
        plasma_surface, 
        d_expand,
        mpol=None,
        ntor=None,
        n_phi=None,
        n_theta=None,
    ):
    
    if mpol is None:
        mpol = plasma_surface.mpol
    if ntor is None:
        ntor = plasma_surface.ntor
    if n_phi is None:
        n_phi = len(plasma_surface.quadpoints_phi) 
    if n_theta is None:
        n_theta = len(plasma_surface.quadpoints_theta) 
    offset_surface = SurfaceRZFourier(
        nfp=plasma_surface.nfp, 
        stellsym=plasma_surface.stellsym, 
        mpol=plasma_surface.mpol,
        ntor=plasma_surface.ntor,
        quadpoints_phi=jnp.linspace(0, 1, n_phi, endpoint=False),
        quadpoints_theta=jnp.linspace(0, 1, n_theta, endpoint=False),
    )
    offset_surface.set_dofs(plasma_surface.to_RZFourier().get_dofs())
    offset_surface.extend_via_normal(d_expand)
    # A naively expanded surface usually has self-intersections 
    # in the poloidal cross section when the plasma is bean-shaped.
    # To avoid this, we create a new surface by taking each poloidal cross section's 
    # convex hull.
    gamma = offset_surface.gamma().copy()
    gamma_R = jnp.linalg.norm([gamma[:,:,0], gamma[:,:,1]], axis=0)
    gamma_Z = gamma[:,:,2]
    gamma_new = jnp.zeros_like(gamma)

    for i_phi in range(gamma.shape[0]):
        phi_i = offset_surface.quadpoints_phi[i_phi]
        cross_sec_R_i = gamma_R[i_phi]
        cross_sec_Z_i = gamma_Z[i_phi]

        # Create periodic array for interpolation
        (
            vertices_R_periodic_i,
            vertices_Z_periodic_i
        ) = callable_process_and_wrap_RZ(cross_sec_R_i, cross_sec_Z_i)

        
        # Temporarily vertically the cross section.
        # For centering the theta=0 curve to be along 
        # the projection of the axis to the outboard side.
        Z_center_i = jnp.average(vertices_Z_periodic_i[:-1])
        vertices_Z_periodic_i = vertices_Z_periodic_i - Z_center_i

        # Parameterize the series of vertices with 
        # arc length
        delta_R = jnp.diff(vertices_R_periodic_i)
        delta_Z = jnp.diff(vertices_Z_periodic_i)
        segment_length = jnp.sqrt(delta_R**2 + delta_Z**2)
        arc_length = jnp.cumsum(segment_length)
        arc_length_periodic = jnp.concatenate(([0], arc_length))
        arc_length_periodic_norm = arc_length_periodic/arc_length_periodic[-1]
        # Interpolate
        spline_i = CubicSpline(
            arc_length_periodic_norm, 
            jnp.array([vertices_R_periodic_i, vertices_Z_periodic_i]).T,
            bc_type='periodic'
        )
        # Calculating the phase shift in quadpoints_theta needed 
        # to center the theta=0 point to the intersection between
        # the Z=0 plane and the ourboard of the winding surface.
        Z_roots = spline_i.roots(extrapolate=False)[-1]
        root_RZ_i = spline_i(Z_roots)
        root_R_i = root_RZ_i[:, 0]
        # Choose the outboard root as theta=0
        phase_shift = Z_roots[jnp.argmax(root_R_i)]
        # Re-calculate R and Z from uniformly spaced theta
        conv_gamma_RZ_i = spline_i(offset_surface.quadpoints_theta + phase_shift)
        conv_gamma_R_i = conv_gamma_RZ_i[:, 0]
        conv_gamma_Z_i = conv_gamma_RZ_i[:, 1]
        # Remove the temporary offset introduced earlier
        conv_gamma_Z_i = conv_gamma_Z_i + Z_center_i
        # Calculate X and Y
        conv_gamma_X_i = conv_gamma_R_i*jnp.cos(phi_i*jnp.pi*2)
        conv_gamma_Y_i = conv_gamma_R_i*jnp.sin(phi_i*jnp.pi*2)
        gamma_new[i_phi, :, 0] = conv_gamma_X_i
        gamma_new[i_phi, :, 1] = conv_gamma_Y_i
        gamma_new[i_phi, :, 2] = conv_gamma_Z_i

    # Fitting to XYZ tensor fourier surface
    winding_surface_new = SurfaceXYZTensorFourier( 
        nfp=offset_surface.nfp,
        stellsym=offset_surface.stellsym,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=offset_surface.quadpoints_phi,
        quadpoints_theta=offset_surface.quadpoints_theta,
    )
    winding_surface_new.least_squares_fit(gamma_new)
    winding_surface_new = winding_surface_new.to_RZFourier()

    # Copying to all field periods
    len_phi_full = len(plasma_surface.quadpoints_phi) * plasma_surface.nfp
    winding_surface_out = SurfaceRZFourier(
        nfp=plasma_surface.nfp, 
        stellsym=plasma_surface.stellsym, 
        mpol=mpol, 
        ntor=ntor, 
        quadpoints_phi=jnp.arange(len_phi_full)/len_phi_full, 
        quadpoints_theta=plasma_surface.quadpoints_theta, 
    )
    winding_surface_out.set_dofs(winding_surface_new.get_dofs())
    return(winding_surface_out)

# Assumes that source_surface only contains 1 field period!
def gen_normal_winding_surface(source_surface, d_expand):
    # Expanding plasma surface to winding surface
    len_phi = len(source_surface.quadpoints_phi)
    len_phi_full_fp = len_phi * source_surface.nfp
    len_theta = len(source_surface.quadpoints_theta)
    
    winding_surface = SurfaceRZFourier(
        nfp=source_surface.nfp, 
        stellsym=source_surface.stellsym, 
        mpol=source_surface.mpol, 
        ntor=source_surface.ntor, 
        quadpoints_phi=jnp.arange(len_phi_full_fp)/len_phi_full_fp, 
        quadpoints_theta=jnp.arange(len_theta)/len_theta, 
    )
    winding_surface.set_dofs(source_surface.get_dofs())
    winding_surface.extend_via_normal(d_expand)
    return(winding_surface)

''' Operator projection '''
def project_arr_coord(
    operator, 
    unit1, unit2, unit3):
    '''
    Project a (n_phi, n_theta, 3, <shape>) array in a given basis (unit1, unit2, unit3) 
    with shape (n_phi, n_theta, 3). 
    Outputs: (n_phi, n_theta, 3, <shape>)
    Sample the first field period when one_field_period is True.
    '''
    # Memorizing shape of the last dimensions of the array
    len_phi = operator.shape[0]
    len_theta = operator.shape[1]
    operator_shape_rest = list(operator.shape[3:])
    operator_reshaped = operator.reshape((len_phi, len_theta, 3, -1))
    # Calculating components
    # shape of operator is 
    # (n_grid_phi, n_grid_theta, 3, n_dof, n_dof)
    # We take the dot product between K and unit vectors.
    operator_1 = jnp.sum(unit1[:,:,:,None]*operator_reshaped, axis=2)
    operator_2 = jnp.sum(unit2[:,:,:,None]*operator_reshaped, axis=2)
    operator_3 = jnp.sum(unit3[:,:,:,None]*operator_reshaped, axis=2)

    operator_1_nfp_recovered = operator_1.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_2_nfp_recovered = operator_2.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_3_nfp_recovered = operator_3.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_comp_arr = jnp.stack([
        operator_1_nfp_recovered,
        operator_2_nfp_recovered,
        operator_3_nfp_recovered
    ], axis=2)
    return(operator_comp_arr)

def project_arr_cylindrical(
        gamma, 
        operator,
    ):
    # Keeping only the x, y components
    r_unit = jnp.zeros_like(gamma)
    r_unit = r_unit.at[:, :, -1].set(0)
    # Calculating the norm and dividing the x, y components by it
    r_unit = r_unit.at[:, :, :-1].set(gamma[:, :, :-1] / jnp.linalg.norm(gamma, axis=2)[:, :, None])

    # Setting Z unit to 1
    z_unit = jnp.zeros_like(gamma)
    z_unit = z_unit.at[:,:,-1].set(1)

    phi_unit = jnp.cross(z_unit, r_unit)
    return(
        project_arr_coord(
            operator,
            unit1=r_unit, 
            unit2=phi_unit, 
            unit3=z_unit,
        )
    )

''' Misc '''
def self_outer_prod_vec(arr_1d):
    '''
    Calculates the outer product of a 1d array with itself.
    '''
    return(arr_1d[:, None]*arr_1d[None,:])

def self_outer_prod_matrix(arr_2d):
    '''
    Calculates the outer product of a matrix with itself. 
    Has the effect (Return)@x@x = (Return)@(xx^T) = (Input@x)(Input@x)^T
    '''
    return(arr_2d[None, :, :, None] * arr_2d[:, None, None, :])

def run_nescoil_legacy(
        filename,
        mpol = 4,
        ntor = 4,
        coil_ntheta_res = 1,
        coil_nzeta_res = 1,
        plasma_ntheta_res = 1,
        plasma_nzeta_res = 1):
    '''
    Loads a CurrentPotentialFourier, a CurrentPotentialSolve 
    and a CurrentPotentialFourier containing the NESCOIL result.
    
    Works for 
    '/simsopt/tests/test_files/regcoil_out.hsx.nc'
    '/simsopt/tests/test_files/regcoil_out.li383.nc'
    '''
    # Load in low-resolution NCSX file from REGCOIL
    
    cp_temp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)
    coil_nzeta_res *= cp_temp.nfp
 
    cpst = CurrentPotentialSolve.from_netcdf(
        filename, plasma_ntheta_res, plasma_nzeta_res, coil_ntheta_res, coil_nzeta_res
    )
    cp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)

    # Overwrite low-resolution file with more mpol and ntor modes
    cp = CurrentPotentialFourier(
        cpst.winding_surface, mpol=mpol, ntor=ntor,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        stellsym=True)
    
    cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma)
    # cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma, cpst.B_GI)
    
    # Discard L2 regularization for testing against linear relaxation
    lambda_reg = 0
    optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
    cp_opt = cpst.current_potential
    
    return(cp, cpst, cp_opt, optimized_phi_mn)

def run_nescoil(
        filename,
        mpol = 4,
        ntor = 4,
        d_expand_norm = 2,
        coil_ntheta_res = 1,
        coil_nzeta_res = 1,
        plasma_ntheta_res = 1,
        plasma_nzeta_res = 1):
    '''
    Loads a CurrentPotentialFourier, a CurrentPotentialSolve 
    and a CurrentPotentialFourier containing the NESCOIL result.
    
    Works for 
    '/simsopt/tests/test_files/regcoil_out.hsx.nc'
    '/simsopt/tests/test_files/regcoil_out.li383.nc'
    '''
    # Load in low-resolution NCSX file from REGCOIL
    
    cp_temp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)
    coil_nzeta_res *= cp_temp.nfp
 
    cpst = CurrentPotentialSolve.from_netcdf(
        filename, plasma_ntheta_res, plasma_nzeta_res, coil_ntheta_res, coil_nzeta_res
    )

    plasma_surface = cpst.plasma_surface
    d_expand = d_expand_norm * plasma_surface.minor_radius()
    winding_surface_conv = gen_conv_winding_surface(plasma_surface, d_expand)

    # Overwrite low-resolution file with more mpol and ntor modes
    cp = CurrentPotentialFourier(
        winding_surface_conv, mpol=mpol, ntor=ntor,
        net_poloidal_current_amperes=cp_temp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp_temp.net_toroidal_current_amperes,
        stellsym=True)
    
    cpst = CurrentPotentialSolve(cp, plasma_surface, cpst.Bnormal_plasma)
    
    # Discard L2 regularization for testing against linear relaxation
    lambda_reg = 0
    optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
    cp_opt = cpst.current_potential
    
    return(cp, cpst, cp_opt, optimized_phi_mn)


def change_cp_resolution(cp: CurrentPotentialFourier, n_phi:int, n_theta:int):
    '''
    Takes a CurrentPotentialFourier, keeps its Fourier
    components but changes the phi (zeta), theta grid numbers.
    '''
    winding_surface_new = SurfaceRZFourier(
        nfp=cp.winding_surface.nfp, 
        stellsym=cp.winding_surface.stellsym, 
        mpol=cp.winding_surface.mpol, 
        ntor=cp.winding_surface.ntor,
        quadpoints_phi=jnp.linspace(0,1,n_phi), 
        quadpoints_theta=jnp.linspace(0,1,n_theta)
    )
    winding_surface_new.set_dofs(cp.winding_surface.get_dofs())

    cp_new = CurrentPotentialFourier(
        winding_surface_new, 
        mpol=cp.mpol, 
        ntor=cp.ntor,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        stellsym=True)
    return(cp_new)

''' 
Conic helper methods
Helpful matrices used during the construction of SDP problems.
'''
# Creates an operator on X=
# Phi Phi^T, Phi
# Phi      , 1
# Tr(OX) = Phi^T A Phi + b Phi + c.
# def quad_operator(A, b, c, dim_phi):

# Trace of (this matrix)@X contains 
# only the n_item-th item in the 
# last row/col of X.
# n_X is the size of X.
def sdp_helper_last_col(n_item, n_X):
    return(sdp_helper_get_elem(n_item, -1, n_X))

# Trace of (this matrix)@X contains 
# only the (a, b) item in the 
# last row/col of X.
# n_X is the size of X.
def sdp_helper_get_elem(a, b, n_X):
    matrix = jnp.zeros((n_X,n_X))
    matrix = matrix.at[a,b].set(1)
    return(matrix)

# This matrix creates an (n+1, n+1) matrix B
# from (n, n) matrix A so that
# tr(B@X) = tr(A@Phi)+p_sign*p
# For defining the inequalities:
# tr(A_ij@Phi)-p <= b_ij
# tr(A_ij@Phi)+p >= b_ij
def sdp_helper_p_inequality(matrix_A, p_sign):
    n_X = jnp.shape(matrix_A)[0]+1
    return(sdp_helper_expand_and_add_diag(matrix_A, -1, p_sign, n_X))

# This matrix creates an (n_X, n_X) matrix B
# from (n, n) matrix A so that
# tr(B@X) = tr(A@Phi)+p_sign*p
# where p is the n_p-th diagonal element.
# For defining the inequalities:
# tr(A_ij@Phi)-p <= b_ij
# tr(A_ij@Phi)+p >= b_ij
def sdp_helper_expand_and_add_diag(matrix_A, n_p, p_sign, n_X):
    n_A = jnp.shape(matrix_A)[0]
    matrix = jnp.zeros((n_X, n_X))
    matrix[:n_A,:n_A]=matrix_A
    matrix[n_p,n_p]=p_sign
    return(matrix)

# This matrix C satisfies
# tr(C@X)=p (p is X's last diagonal element)
# n_X is the size of X
def sdp_helper_p(n_X):
    return(sdp_helper_get_elem(-1, -1, n_X))

'''
Analysis helper methods
'''
def last_exact_i_X_list(X_value_list, theshold=1e-5):
    '''
    Obtaining a list of second greatest eigenvalue
    and find the index of the last item where the
    eigenvalue <= threshold
    '''
    second_max_eig_list = []
    eig_list=[]
    for i in range(len(X_value_list)):
        eigvals, _ = jnp.linalg.eig(X_value_list[i][:-1, :-1])
        eig_list.append(eigvals)
        second_max_eig_list.append(jnp.sort(jnp.abs(eigvals))[-2])
    last_exact_i = jnp.argwhere(
        jnp.max(jnp.abs(eig_list)[:, 1:], axis=1)<theshold
    ).flatten()[-1]
    return(last_exact_i)

def find_most_similar(Phi_list, f, f_value):
    '''
    Loop over a list of arrays, call a function f over its elements, 
    and find the index of Phi with the closest f(Phi) to f_value
    '''
    min_f_diff = float('inf')
    most_similar_index = 0
    for i_case in range(len(Phi_list)):
        Phi_l2_i = Phi_list[i_case]
        f_i = f(Phi_l2_i)
        f_diff = jnp.abs(f_i - f_value)
        if f_diff<min_f_diff:
            most_similar_index = i_case
            min_f_diff = f_diff
    return(most_similar_index)

def shallow_copy_cp_and_set_dofs(cp, dofs):
    '''
    Shallow copy a CurrentPotential and set new DOF.
    Note that the winding_surface is still the 
    same instance and will change when the original's
    is modified.
    '''
    cp_new = CurrentPotentialFourier(
        cp.winding_surface, 
        cp.net_poloidal_current_amperes,
        cp.net_toroidal_current_amperes, 
        cp.nfp, 
        cp.stellsym,
        cp.mpol, 
        cp.ntor,
    )
    cp_new.set_dofs(dofs) # 28 for w7x
    return(cp_new)
