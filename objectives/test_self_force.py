import sys
import os 
sys.path.insert(0, './robin_volpe_src')
import time 
sys.path.insert(0, '../build')
sys.path.insert(0, '..')
import biest_call
from matplotlib import pyplot as plt

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

# Importing
from regcoil import *
import numpy as np
import toroidal_surface
import avg_laplace_force
from vector_field_on_TS import *
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve, CurrentPotential
from self_force_operator import self_force_integrand_nominator_cylindrical
from utils import gen_conv_winding_surface, project_arr_cylindrical
from f_b_and_k_operators import f_B_operator_and_current_scale

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
# -12: precise QA DOESN'T WORK
surface_i = -12
num_quadpoints_phi = 32 # 32 # 64 x 64 doesnt work!!!!!
num_quadpoints_theta = 32 # 32 
mpol_coil  = 2
ntor_coil  = 2
lamb2 = 2.5e-16

# Loading plasma surface
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
nzeta_plasma = num_quadpoints_phi + 1 
ntheta_plasma = num_quadpoints_theta + 1 
nzeta_coil = num_quadpoints_phi + 1 
ntheta_coil = num_quadpoints_theta + 1

# Loading surface
plasma_surface = SurfaceRZFourier(
    nfp=nfp, 
    stellsym=True, 
    mpol=surf_dict['mpol'], 
    ntor=surf_dict['ntor'], 
    quadpoints_phi=np.linspace(0, 1/nfp, num_quadpoints_phi),
    quadpoints_theta=np.linspace(0, 1, num_quadpoints_theta),
)
plasma_surface.set_dofs(surf_dict['dofs'])

winding_surface = gen_conv_winding_surface(plasma_surface, 0.5 * plasma_surface.minor_radius())
cp = CurrentPotentialFourier(
    winding_surface, mpol=mpol_coil, ntor=ntor_coil,
    net_poloidal_current_amperes=net_poloidal_current_Amperes,
    net_toroidal_current_amperes=net_toroidal_current_Amperes,
    stellsym=True)
cpst = CurrentPotentialSolve(cp, plasma_surface, 0)
(
    f_B_x_operator, 
    B_normal, 
    current_scale, 
    f_B_scale
) = f_B_operator_and_current_scale(cpst, normalize=True)

surface_to_robin_txt(plasma_surface, 'robin_volpe_tmp/plasma_surf.txt')
surface_to_robin_txt(winding_surface, 'robin_volpe_tmp/cws.txt')


''' Calculating self force using Robin's code '''


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

# Finding regcoil sln
coeff_l,div_free=Regcoil(G,I,lamb2,phisize,cws,plasma_surf,path_bnorm=path_bnorm,curpol=curpol)

# Calculating force
eps=1e-2
lst_theta=range(ntheta_coil-1)
lst_zeta=range(nzeta_coil-1)
Pm=(np.array([cws.X,cws.Y,cws.Z])+eps*cws.n)[:,:-1,:-1].reshape((3,-1)).transpose()
Pp=(np.array([cws.X,cws.Y,cws.Z])-eps*cws.n)[:,:-1,:-1].reshape((3,-1)).transpose()
# Exact
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
# Epsilon
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


''' Converting REGCOIL sln from Robin's code to simsopt '''


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
        cp.set_phis(m, -n, -sol_S[i, j])
# QUADCOIL vars
unscaled_small_x = np.concatenate([cp.get_dofs(), [1]])
unscaled_x = unscaled_small_x[:, None] * unscaled_small_x[None, :]
scaled_small_x = np.concatenate([current_scale * cp.get_dofs(), [1]])
scaled_x = scaled_small_x[:, None] * scaled_small_x[None, :]


'''Self force operator'''


def self_force_operator_cylindrical(
        cp_x:CurrentPotentialFourier, current_scale):
    '''
    First, we need a temporary cp, cp_eval with a cp_eval.winding_surface
    containing only 1 field period to reduce array sizes by nfp times.
    BIEST seems to only support evaluating int f(x, y)da(x) 
    using the same grid for x and y. If this isn't the case, then 
    we need to have different CurrentPotential and Surface instances 
    for x and y.
    '''
    # Convert the surface type
    winding_surface_original_RZ = cp_x.winding_surface.to_RZFourier()
    len_phi = len(winding_surface_original_RZ.quadpoints_phi)
    print('Original quadpoint phi num', len(winding_surface_original_RZ.quadpoints_phi))
    print('nfp', nfp)
    print('quadpt mod', len_phi % nfp)
    if len_phi % nfp != 0:
        raise ValueError(
            'The number of phi quadrature points of cp_x.winding_surface'\
            ' is not an exact multiple of nfp.'
        )
    print()

    # We assume the original winding_surface contains all field periods.
    # Copy the surface and limits the 
    winding_surface_1fp = SurfaceRZFourier( 
        nfp=winding_surface_original_RZ.nfp,
        stellsym=winding_surface_original_RZ.stellsym,
        mpol=winding_surface_original_RZ.mpol,
        ntor=winding_surface_original_RZ.ntor,
        # quadpoints_phi=np.arange(nzeta_coil)/nzeta_coil/3,
        quadpoints_phi=winding_surface_original_RZ.quadpoints_phi[:len_phi//nfp],
        # quadpoints_theta=np.arange(ntheta_coil)/ntheta_coil,
        quadpoints_theta=winding_surface_original_RZ.quadpoints_theta,
    )
    winding_surface_1fp.set_dofs(winding_surface_original_RZ.get_dofs())
    print('Winding surface for evaluation:')
    print('len_phi', len_phi)
    print('new len_phi', len(winding_surface_1fp.quadpoints_phi))
    winding_surface_original_RZ.plot()
    winding_surface_1fp.plot()

    # Define a temporary CurrerntPotentialFourier object using this object
    cp_eval = CurrentPotentialFourier(
        winding_surface_1fp, mpol=cp_x.mpol, ntor=cp_x.ntor,
        net_poloidal_current_amperes=cp_x.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp_x.net_toroidal_current_amperes,
        stellsym=cp_x.stellsym
    )
    # No need to copy cp_x's dof because it'll not be used.
    
    (
        func_single_cylindrical, func_double_cylindrical,
        K_op_cylindrical,
    ) = self_force_integrand_nominator_cylindrical(cp_eval, current_scale)
    # Reshaping for using with BIEST
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
    plt.title('func_double_cylindrical_reshaped[:,:,20]')
    plt.pcolor(func_double_cylindrical_reshaped[:,:,20])
    plt.show()
    # Mutation for the c++ function biest_call.integrate_multi
    result_single = np.zeros_like(func_single_cylindrical_reshaped)
    result_double = np.zeros_like(func_double_cylindrical_reshaped)

    # Calling BIEST, the singular integral code.
    time1 = time.time()
    biest_call.integrate_multi(
        cp_eval.winding_surface.gamma(), # xt::pyarray<double> &gamma,
        func_single_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
        result_single, # xt::pyarray<double> &result,
        True, # bool, whether to use the single layer kernel
        10, # int digits,
        cp_eval.nfp, # int nfp
        True # Whether to detect sign flip for a double-layer kernel.  
    )
    biest_call.integrate_multi(
        cp_eval.winding_surface.gamma(), # xt::pyarray<double> &gamma,
        func_double_cylindrical_reshaped, # xt::pyarray<double> &func_in_single,
        result_double, # xt::pyarray<double> &result,
        False, # bool, whether to use the single layer kernel
        10, # int digits,
        cp_eval.nfp, # int nfp
        True # Whether to detect sign flip for a double-layer kernel.  
    )
    time2 = time.time()
    num_integral = func_double_cylindrical_reshaped.shape[-1]*2# The total number of BIEST integrals evaluated
    print('BIEST run completed.')
    print('Total time(s):               ', time2-time1)
    print('Total number of integrals:   ', num_integral)
    print('Average time per integral(s):', (time2-time1)/num_integral)


    plt.title('results_double')
    plt.pcolor(result_double[:,:,100])
    plt.colorbar()
    plt.show()

    plt.title('results_single')
    plt.pcolor(result_single[:,:,100])
    plt.colorbar()
    plt.show()
    # BIEST's convention has a 1/4pi in front of the kernel.
    # We remove it for consistency because we are Robin's code does not.
    result_tot = 4 * np.pi * (result_single + result_double).reshape(
        result_single.shape[0],
        result_single.shape[1],
        3, 3, -1
    )
    self_force_operator = np.sum(
        K_op_cylindrical[:, :, :, None, :, None] 
        * result_tot[:, :, :, :, None, :],
        axis = 2
    )
    
    return(
        func_single_cylindrical,
        func_double_cylindrical,
        K_op_cylindrical,
        result_single,
        result_double,
        self_force_operator)

(
    func_single_cylindrical,
    func_double_cylindrical,
    K_op_cylindrical,
    result_single,
    result_double,
    self_force_op
) = self_force_operator_cylindrical(cp, 1)
(
    func_single_cylindrical_scaled,
    func_double_cylindrical_scaled,
    K_op_cylindrical_scaled,
    result_single_scaled,
    result_double_scaled,
    self_force_op_scaled
) = self_force_operator_cylindrical(cp, current_scale)

self_force_op = self_force_operator_cylindrical(cp, 1)
self_force_op_scaled = self_force_operator_cylindrical(cp, current_scale)


L_quadcoil = np.trace(self_force_op @ unscaled_x, axis1=-1, axis2=-2)
L_quadcoil2 = np.trace(self_force_op_scaled @ scaled_x, axis1=-1, axis2=-2)

winding_surface_1fp = SurfaceRZFourier( 
        nfp=winding_surface.nfp,
        stellsym=winding_surface.stellsym,
        mpol=winding_surface.mpol,
        ntor=winding_surface.ntor,
        # quadpoints_phi=np.arange(nzeta_coil)/nzeta_coil/3,
        quadpoints_phi=np.linspace(0, 1/cp.nfp, num_quadpoints_phi, endpoint=False),
        # quadpoints_theta=np.arange(ntheta_coil)/ntheta_coil,
        quadpoints_theta=np.linspace(0, 1, num_quadpoints_theta, endpoint=False),
    )
winding_surface_1fp.set_dofs(winding_surface.get_dofs())
L_eps_cylindrial = project_arr_cylindrical(
    winding_surface_1fp, 
    L_eps.swapaxes(0, 1)
)
L_robin_cylindrical = project_arr_cylindrical(
    winding_surface_1fp, 
    laplace_array.swapaxes(0, 1)
)
plt.title('L-eps from Robin-Volpe')
plot_vec(L_eps_cylindrial)
plt.title('L from Robin-Volpe')
plot_vec(L_robin_cylindrical)
plt.title('L from our code')
plot_vec(L_quadcoil)
plt.title('L from our code, with scaling')
plot_vec(L_quadcoil2)