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
from self_force_operator import self_force_operator_cylindrical, self_force_integrand_nominator_cylindrical
from utils import gen_conv_winding_surface, project_arr_cylindrical
from f_b_and_k_operators import f_B_operator_and_current_scale


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
# -12: precise QA

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

# Loading surface
plasma_surface = SurfaceRZFourier(
    nfp=nfp, 
    stellsym=True, 
    mpol=surf_dict['mpol'], 
    ntor=surf_dict['ntor'], 
    quadpoints_phi=np.linspace(0, 1/nfp, len_phi),
    quadpoints_theta=np.linspace(0, 1, len_theta),
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
scaled_small_x = np.concatenate([cp.get_dofs(), [1]])
scaled_x = scaled_small_x[:, None] * scaled_small_x[None, :]


scaled_small_x2 = np.concatenate([current_scale * cp.get_dofs(), [1]])
scaled_x2 = scaled_small_x2[:, None] * scaled_small_x2[None, :]


self_force_op = self_force_operator_cylindrical(cp, 1)
self_force_op_scaled = self_force_operator_cylindrical(cp, current_scale)


L_quadcoil = np.trace(self_force_op @ scaled_x, axis1=-1, axis2=-2)
L_quadcoil2 = np.trace(self_force_op_scaled @ scaled_x2, axis1=-1, axis2=-2)

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
winding_surface_1fp = SurfaceRZFourier( 
        nfp=winding_surface.nfp,
        stellsym=winding_surface.stellsym,
        mpol=winding_surface.mpol,
        ntor=winding_surface.ntor,
        # quadpoints_phi=np.arange(nzeta_coil)/nzeta_coil/3,
        quadpoints_phi=np.linspace(0, 1/cp.nfp, len_phi, endpoint=False),
        # quadpoints_theta=np.arange(ntheta_coil)/ntheta_coil,
        quadpoints_theta=np.linspace(0, 1, len_theta, endpoint=False),
    )
winding_surface_1fp.set_dofs(winding_surface.get_dofs())
L_eps_cylindrial =  project_arr_cylindrical(
    winding_surface_1fp, 
    L_eps.swapaxes(0, 1)
)
L_robin_cylindrical =  project_arr_cylindrical(
    winding_surface_1fp, 
    laplace_array.swapaxes(0, 1)
)
print('L-eps from Robin-Volpe')
plot_vec(L_eps_cylindrial)
print('L from Robin-Volpe')
plot_vec(L_robin_cylindrical)
print('L from our code')
plot_vec(L_quadcoil)
print('L from our code, with scaling')
plot_vec(L_quadcoil2)


# Estimating the missing scalar factor
def f_scalar_factor(x):
    return(np.linalg.norm(L_robin_cylindrical * x - L_quadcoil))

from scipy.optimize import minimize
res = minimize(f_scalar_factor, 10)
print('An estimate for the missing constant factor is:', res.x)
print('nfp is', nfp)
print('L2 error', res.fun)

np.linalg.norm(L_eps_cylindrial - L_robin_cylindrical)

