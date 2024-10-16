import sys
import os 
sys.path.insert(0, './robin_volpe_src')
import time 
sys.path.insert(0, '../build')
sys.path.insert(0, '..')
import biest_call
from regcoil import *
import numpy as np
import toroidal_surface
import avg_laplace_force
from vector_field_on_TS import *
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve, CurrentPotential

from operator_helper import (
    norm_helper, Kdash_helper, 
    unitnormaldash, grad_helper
)
from f_b_and_k_operators import AK_helper
from utils import gen_conv_winding_surface, project_arr_cylindrical
# Loads the data as a simsopt surface. Assumes stellarator symmetry. 
# For testing the self-force calculation.
def load_robin_txt_to_simsopt_surface(path, nfp, quadpoints_phi, quadpoints_theta):
    # Load the data from the .txt file into a numpy array
    data = np.loadtxt(path)

    # Extract each column into a separate 1-D array
    m_robin = data[:, 0].astype(int)   # First column
    n_robin = data[:, 1].astype(int)   # Second column
    Rmn_robin = data[:, 2]             # Third column
    Zmn_robin = data[:, 3]             # Fourth column

    surface_out = SurfaceRZFourier( 
        nfp=nfp,
        stellsym=True,
        mpol=int(np.max(m_robin)),
        ntor=int(np.max(np.abs(n_robin))),
        # quadpoints_phi=np.arange(nzeta_coil)/nzeta_coil/3,
        quadpoints_phi=quadpoints_phi,
        # quadpoints_theta=np.arange(ntheta_coil)/ntheta_coil,
        quadpoints_theta=quadpoints_theta,
    )
    for i in range(len(m_robin)):
        m = m_robin[i]
        n = - n_robin[i] # Different coordinate convention! Refer to robin_volpe_li383/fourier_descr.txt
        surface_out.set_rc(m, n, Rmn_robin[i])
        surface_out.set_zs(m, n, Zmn_robin[i])
    
    return(surface_out)

# Converts a surface with stellarator symmetry to .txt files in Robin-Volpe's 
# convention
def surface_to_robin_txt(surface, dir_out):
    # Combine the arrays into a 2D array with columns
    m_robin = []
    n_robin = []
    Rmn_robin = []
    Zmn_robin = []
    for m in np.arange(surface.mpol+1):
        for n in np.arange(surface.ntor*2+1)-surface.ntor:
            m_robin.append(m)
            n_robin.append(-n)
            Rmn_robin.append(surface.get_rc(m, n))
            Zmn_robin.append(surface.get_zs(m, n))
    data = np.column_stack((np.array(m_robin), np.array(n_robin), np.array(Rmn_robin), np.array(Zmn_robin)))
    # Write the data to a text file with the specified format
    np.savetxt(dir_out, data, fmt='%d    %d  %.4E  %.4E')
    # Configurations
# 11: LHD
# -6: w7x

surface_i = -12
len_phi = 64 # 32
len_theta = 48 # 32 
mpol_coil  = 2
ntor_coil  = 2
lamb2 = 2.5e-16
path_plasma_surface = '../test_data/plasma_surface/'
list_plasma_surface = os.listdir(path_plasma_surface)
current_plasma_surface = list_plasma_surface[surface_i]
print('Currently testing with:', current_plasma_surface)

# Loading surface
# This test code is only compatible with stellarator symmetric cases!
surf_dict = np.load(path_plasma_surface + current_plasma_surface, allow_pickle=True).item()
assert(surf_dict['stellsym'])

# Robin-Volpe configurations
nfp = surf_dict['nfp']
# Note: Robin-Volpe's code divide the net poloidal current by nfp.
# Search for the following line:
# G, I = net_poloidal_current_Amperes/nfp, net_toroidal_current_Amperes
net_poloidal_current_Amperes = surf_dict['net_poloidal_current']
net_toroidal_current_Amperes = 0.
curpol=0 # 4.9782004309255496 for li383. only useful when there's b_norm. We set it to 0.
gamma=lamb2
nzeta_plasma = len_phi + 1 
ntheta_plasma = len_theta + 1 
nzeta_coil = len_phi + 1 
ntheta_coil = len_theta + 1

# winding_surface_test_terms = load_robin_txt_to_simsopt_surface(
#     path=path_cws, 
#     nfp=3, 
#     # The minus 2 makes sure the grids are aligned correctly
#     quadpoints_phi=np.linspace(0, 1, nzeta_coil*3-2), 
#     # quadpoints_theta=np.arange(ntheta_coil)/ntheta_coil,
#     quadpoints_theta=np.linspace(0, 1, ntheta_coil),
# )
# Loading surface
plasma_surface = SurfaceRZFourier(
    nfp=nfp, 
    stellsym=True, 
    mpol=surf_dict['mpol'], 
    ntor=surf_dict['ntor'], 
    quadpoints_phi=np.linspace(0, 1/nfp, nzeta_plasma),
    quadpoints_theta=np.linspace(0, 1, ntheta_plasma),
)
plasma_surface.set_dofs(surf_dict['dofs'])

winding_surface_original = gen_conv_winding_surface(plasma_surface, 0.5 * plasma_surface.minor_radius())
# Changing quadpoints so that it aligns with the quadpoints 
# in Robin's code
winding_surface_test_terms = SurfaceRZFourier(
    nfp=nfp, 
    stellsym=True, 
    mpol=winding_surface_original.mpol, 
    ntor=winding_surface_original.ntor, 
    quadpoints_phi=np.linspace(0, 1, nzeta_coil*nfp-2), 
    quadpoints_theta=np.linspace(0, 1, ntheta_coil),
)
winding_surface_test_terms.set_dofs(winding_surface_original.get_dofs())
surface_to_robin_txt(plasma_surface, 'robin_volpe_tmp/plasma_surf.txt')
surface_to_robin_txt(winding_surface_test_terms, 'robin_volpe_tmp/cws.txt')
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

# Seem to calculate the components of lorentz force
def compute_aux(cws,laplace_array,sel=1):
    laplace_norm=np.linalg.norm(laplace_array,axis=2)
    laplace_normal=np.sum(-1*cws.normal[:-1:sel,:-1:sel,:]*laplace_array,axis=2)
    laplace_tangent=laplace_array-laplace_normal[:,:,np.newaxis]*(-1*cws.normal[:-1:sel,:-1:sel,:])
    laplace_norm_tangent=np.linalg.norm(laplace_tangent,axis=2)
    return [laplace_norm,laplace_normal,laplace_tangent,laplace_norm_tangent]

phisize=(ntor_coil,mpol_coil)
G, I = net_poloidal_current_Amperes/nfp, net_toroidal_current_Amperes
path_plasma='robin_volpe_tmp/plasma_surf.txt'
path_cws='robin_volpe_tmp/cws.txt'
path_bnorm='robin_volpe_tmp/bnorm.txt'
plasma_surf=toroidal_surface.Toroidal_surface(W7x_pathfile=path_plasma,nbpts=(ntheta_plasma,nzeta_plasma),Np=nfp)
cws=toroidal_surface.Toroidal_surface(W7x_pathfile=path_cws,nbpts=(ntheta_coil,nzeta_coil),Np=nfp)
div_free=vector_field_on_TS.Div_free_vector_field_on_TS(cws)
avg=avg_laplace_force.Avg_laplace_force(cws)


coeff_l,div_free=Regcoil(G,I,lamb2,phisize,cws,plasma_surf,path_bnorm=path_bnorm,curpol=curpol)
eps=1e-2
lst_theta=range(ntheta_coil-1)
lst_zeta=range(nzeta_coil-1)
Pm=(np.array([cws.X,cws.Y,cws.Z])+eps*cws.n)[:,:-1,:-1].reshape((3,-1)).transpose()
Pp=(np.array([cws.X,cws.Y,cws.Z])-eps*cws.n)[:,:-1,:-1].reshape((3,-1)).transpose()

avg=avg_laplace_force.Avg_laplace_force(cws)
sol_C,sol_S=vector_field_on_TS.Div_free_vector_field_on_TS.array_coeff_to_CS(coeff_l,phisize)
coeff=(G,I,sol_C,sol_S)
laplace_array=avg.f_laplace_optimized(ntheta_coil-1,nzeta_coil-1,coeff,coeff)
laplace_aux=compute_aux(cws,laplace_array)#[laplace_norm,laplace_normal,laplace_tangent,laplace_norm_tangent]
Bm=div_free.compute_B(np.array(Pm),coeff)
Bp=div_free.compute_B(np.array(Pp),coeff)
j1=div_free.get_full_j(coeff)
L_eps=np.zeros((len(lst_theta),len(lst_zeta),3))
B_eps=np.zeros((len(lst_theta),len(lst_zeta),3))
flag=0
for theta in lst_theta:
    for zeta in lst_zeta:
        L_eps[flag//len(lst_zeta),flag%len(lst_zeta),:]=0.5*(np.cross(j1[theta,zeta,:],Bm[:,flag]+Bp[:,flag]))
        B_eps[flag//len(lst_zeta),flag%len(lst_zeta),:]=0.5*(Bm[:,flag]+Bp[:,flag])
        flag+=1
L_eps_norm=np.linalg.norm(L_eps,axis=2)
B_eps_norm=np.linalg.norm(B_eps,axis=2)
j_norm=np.linalg.norm(j1[:-1,:-1,:],axis=2)
j_scalar_B=np.sum(B_eps*j1[:-1,:-1,:],axis=2)
acos=np.arccos(j_scalar_B/(B_eps_norm*j_norm))
B_eps_normal=np.sum(-1*cws.normal[:-1,:-1,:]*B_eps,axis=2)

cp_test_terms = CurrentPotentialFourier(
    winding_surface_test_terms, mpol=mpol_coil, ntor=ntor_coil,
    net_poloidal_current_amperes=net_poloidal_current_Amperes,
    net_toroidal_current_amperes=net_toroidal_current_Amperes,
    stellsym=True)
current_scale = 1 # 2e-7
sol_C, sol_S = Div_free_vector_field_on_TS.array_coeff_to_CS(
    coeff_l,
    phisize
)

# Converting the current potential harmonics into simsopt. 
# Some sign and indexing conventions are different.
for i in np.arange(sol_S.shape[0]):
    for j in np.arange(sol_S.shape[1]):
        m = i
        n = j-(sol_S.shape[1]-1)//2
        cp_test_terms.set_phis(m, -n, -sol_S[i, j])
scaled_small_x = np.concatenate([current_scale * cp_test_terms.get_dofs(), [1]])
scaled_x = scaled_small_x[:, None] * scaled_small_x[None, :]

def self_force_integrand_nominator_cylindrical(
        cp_x:CurrentPotential, current_scale, 
    ):
    ''' 
    Calculates the nominators of the sheet current self-force in Robin, Volpe 2022.
    The K_y dependence is lifted outside the integrals. Therefore, the nominator 
    this function calculates are operators that acts on the QUADCOIL vector
    (scaled Phi, 1). The operator produces a 
    (n_phi_x, n_theta_x, 3(xyz, to act on Ky), 3(xyz), n_dof+1)
    After the integral, this will become a (n_phi_y, n_theta_y, 3, 3, n_dof+1)
    tensor that acts on K(y) to produce a vector with shape (n_phi_y, n_theta_y, 3, n_dof+1, n_dof+1)
    Shape: (n_phi_x, n_theta_x, 3(xyz), 3(xyz), n_dof+1(x)).

    Reminder: Do not use this with BIEST, because the x, y, z components of the vector field 
    has only one period, however many field periods that vector field has.
    ''' 
    # cp_x = cp_test_terms
    winding_surface_x = cp_x.winding_surface

    unitnormaldash1_x, unitnormaldash2_x = unitnormaldash(winding_surface_x)
    # An assortment of useful quantities related to K
    (
        Kdash1_sv_op_x, 
        Kdash2_sv_op_x, 
        Kdash1_const_x,
        Kdash2_const_x
    ) = Kdash_helper(cp_x, current_scale)

    # Operators that acts on the quadcoil vector.
    # Shape: (n_phi_x, n_theta_x, 3, n_dof+1)
    Kdash1_op_x = np.concatenate([
        Kdash1_sv_op_x,
        Kdash1_const_x[:, :, :, None]
    ], axis=-1)
    Kdash2_op_x = np.concatenate([
        Kdash2_sv_op_x,
        Kdash2_const_x[:, :, :, None]
    ], axis=-1)

    # Contravariant basis of x
    grad1_x, grad2_x = grad_helper(winding_surface_x)
    # Unit normal for x
    unitnormal_x = winding_surface_x.unitnormal()
        
    # The Kx operator, acts on the current potential harmonics (Phi).
    # Shape: (n_phi_x, n_theta_x, 3, n_dof), (n_phi_x, n_theta_x, 3)
    AK_x, bK_x = AK_helper(cp_x)
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

    ''' div_x pi_x '''
    # Shape: (n_phi_x, n_theta_x, 3)
    n_x_dot_grad_n_x = (
        np.sum(unitnormal_x * grad1_x, axis=-1)[:, :, None] * unitnormaldash1_x
        + np.sum(unitnormal_x * grad2_x, axis=-1)[:, :, None] * unitnormaldash2_x
    )
    # Shape: (n_phi_x, n_theta_x, 3)
    div_pi_x = -(
        div_n_x[:, :, None] * unitnormal_x
        + n_x_dot_grad_n_x
    )
    # print('div_pi_x', div_pi_x)
    ''' 
    Integrands

    Integrands in the expression are sorted by their kernel and summed.
    '''
    # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz, resulting components), n_dof+1(x))
    const = 1e-7 
    func_single_xyz = const * (
        (
            unitnormal_x[:, :, :, None, None] 
            * div_n_x[:, :, None, None, None] 
            * K_x_op[:, :, None, :, :]
        ) # Term 1a
        + (
            - grad1_x[:, :, :, None, None] * Kdash1_op_x[:, :, None, :, :]
            - grad2_x[:, :, :, None, None] * Kdash2_op_x[:, :, None, :, :]
        ) # Term 1b
        + (
            K_x_op[:, :, :, None, :] * div_pi_x[:, :, None, :, None]
        ) # Term 3a
        + (
            Kdash1_op_x[:, :, :, None, :] * grad1_x[:, :, None, :, None]
            + Kdash2_op_x[:, :, :, None, :] * grad2_x[:, :, None, :, None]
        ) # Term 3b
    ) 
    # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz, resulting components), n_dof+1(x))
    func_double_xyz = const * (
        (unitnormal_x[:, :, :, None, None] * K_x_op[:, :, None, :, :]) # Term 2
        + (-K_x_op[:, :, :, None, :] * unitnormal_x[:, :, None, :, None]) # Term 4
    )

    A_1a_xyz = (
        unitnormal_x[:, :, :, None, None] 
        * div_n_x[:, :, None, None, None] 
        * K_x_op[:, :, None, :, :]
    )

    # We must perform the xyz -> R,Phi,Z coordinate change twice for both
    # axis 2 and 3. Otherwise the operator will not have 
    # the same nfp-fold discrete symmetry as the equilibrium. 
    func_single_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        func_single_xyz
    )
    func_double_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        func_double_xyz
    )
    # The projection function assumes that the first 3 components of the array represents the 
    # phi, theta grid and resulting components of the array. Hence the swapaxes.
    func_single_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        func_single_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)
    func_double_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        func_double_cylindrical.swapaxes(2, 3) 
    ).swapaxes(2,3)

    K_op_cylindrical = project_arr_cylindrical(
        winding_surface_x, 
        K_x_op
    )
    # Shape: (n_phi_x, n_theta_x, 3(R, Phi Z, acts on K(y)), 3(R, Phi, Z, resulting components), n_dof+1(x))
    return(
        A_1a_xyz,
        func_single_xyz, func_double_xyz,
        K_x_op,
        func_single_cylindrical, func_double_cylindrical,
        K_op_cylindrical,
        Kdash1_op_x,
        Kdash2_op_x,
        div_n_x,
        grad1_x,
        grad2_x,
        div_pi_x,
    )

winding_surface_eval = SurfaceRZFourier( 
    nfp=nfp,
    stellsym=True,
    mpol=winding_surface_test_terms.mpol,
    ntor=winding_surface_test_terms.ntor,
    # quadpoints_phi=np.arange(nzeta_coil)/nzeta_coil/3,
    quadpoints_phi=np.linspace(0, 1/nfp, len_phi, endpoint=False),
    # quadpoints_theta=np.arange(ntheta_coil)/ntheta_coil,
    quadpoints_theta=np.linspace(0, 1, len_theta, endpoint=False),
)
winding_surface_eval.set_dofs(winding_surface_original.get_dofs())
cp_eval = CurrentPotentialFourier(
    winding_surface_eval, mpol=mpol_coil, ntor=ntor_coil,
    net_poloidal_current_amperes=net_poloidal_current_Amperes,
    net_toroidal_current_amperes=net_toroidal_current_Amperes,
    stellsym=True
)
cp_eval.set_dofs(cp_test_terms.get_dofs())

(
    A_1a_xyz,
    func_single_xyz, func_double_xyz,
    K_x_op,
    func_single_cylindrical, func_double_cylindrical,
    K_op_cylindrical,
    Kdash1_op_x_test,
    Kdash2_op_x_test,
    div_n_x_test,
    grad1_x_test,
    grad2_x_test,
    div_pi_x_test,
) = self_force_integrand_nominator_cylindrical(
cp_x = cp_eval, 
current_scale = current_scale, 
)
print('Done.')

def plot_vec(vec):
    plt.figure(figsize=(9,2))
    plt.subplot(1, 3, 1)
    plt.pcolor(vec[:, :, 0])
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.pcolor(vec[:, :, 1])
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.pcolor(vec[:, :, 2])
    plt.colorbar()
    plt.show()
def compare_2(veca, vecb):
    plot_vec(veca)
    plot_vec(vecb)

func_single_cylindrical_reshaped = func_single_cylindrical.reshape((
    func_single_cylindrical.shape[0],
    func_single_cylindrical.shape[1],
    -1
))
func_double_cylindrical_reshaped = func_double_cylindrical.reshape((
    func_double_cylindrical.shape[0],
    func_double_cylindrical.shape[1],
    -1
))
result_single = np.zeros_like(func_single_cylindrical_reshaped)
result_double = np.zeros_like(func_double_cylindrical_reshaped)
biest_call.integrate_multi(
    winding_surface_eval.gamma(), # xt::pyarray<double> &gamma,
    func_single_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
    result_single, # xt::pyarray<double> &result,
    True,
    10, # int digits,
    nfp, # int nfp
    True # Whether to detect sign flip
)
biest_call.integrate_multi(
    winding_surface_eval.gamma(), # xt::pyarray<double> &gamma,
    func_double_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
    result_double, # xt::pyarray<double> &result,
    False,
    10, # int digits,
    nfp, # int nfp
    True # Whether to detect sign flip
)
# BIEST's convention has an extra 1/4pi
result_single = 4 * np.pi * result_single.reshape(
    result_single.shape[0],
    result_single.shape[1],
    3, 3, -1
)
# BIEST's convention has an extra 1/4pi
result_double = 4 * np.pi * result_double.reshape(
    result_double.shape[0],
    result_double.shape[1],
    3, 3, -1
)
single_operator = np.sum(
    K_op_cylindrical[:, :, :, None, :, None] 
    * result_single[:, :, :, :, None, :],
    axis = 2
)
double_operator = np.sum(
    K_op_cylindrical[:, :, :, None, :, None] 
    * result_double[:, :, :, :, None, :],
    axis = 2
)

test_single = (single_operator @ scaled_small_x) @ scaled_small_x
test_double = (double_operator @ scaled_small_x) @ scaled_small_x

laplace_array_single, laplace_array_double = avg.f_laplace_optimized_layer(ntheta_coil-1,nzeta_coil-1,coeff,coeff)
laplace_array_single = laplace_array_single.swapaxes(0, 1)
laplace_array_double = laplace_array_double.swapaxes(0, 1)

laplace_array_single_cylindrial =  project_arr_cylindrical(
    winding_surface_eval, 
    laplace_array_single
)
laplace_array_double_cylindrial =  project_arr_cylindrical(
    winding_surface_eval, 
    laplace_array_double
)
L_eps_cylindrial =  project_arr_cylindrical(
    winding_surface_eval, 
    L_eps.swapaxes(0, 1)
)
L_robin_cylindrical =  project_arr_cylindrical(
    winding_surface_eval, 
    laplace_array.swapaxes(0, 1)
)

print('!!!!!!!! The sign of double terms are incorrect! did BIEST detect sign flip??')

compare_2(laplace_array_single_cylindrial, test_single)
compare_2(laplace_array_double_cylindrial, test_double)