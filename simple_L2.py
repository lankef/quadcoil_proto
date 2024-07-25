from matplotlib import pyplot as plt
import numpy as np

from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve

# Looking into what the format of phi is:
def run_target_test(
    filename = '/mnt/d/Codes/simsopt/tests/test_files/regcoil_out.hsx.nc', 
    lambda_reg
):
    """
        Run REGCOIL (L2 regularization) and Lasso (L1 regularization)
        starting from high regularization to low. When fB < fB_target
        is achieved, the algorithms quit. This allows one to compare
        L2 and L1 results at comparable levels of fB, which seems
        like the fairest way to compare them.
    """

    fB_target = 1e-2
    mpol = 4
    ntor = 4
    coil_ntheta_res = 1
    coil_nzeta_res = coil_ntheta_res
    plasma_ntheta_res = coil_ntheta_res
    plasma_nzeta_res = coil_ntheta_res

    # Load in low-resolution NCSX file from REGCOIL
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
    cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma, cpst.B_GI)

    optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
    cp_opt = cpst.current_potential
    return(cp_opt, cp, cpst, optimized_phi_mn, f_B)


cp_opt, cp, cpst, optimized_phi_mn, f_B = run_target_test(lambda_reg=0.1)

theta_study1d, phi_study1d = cp_opt.quadpoints_theta, cp_opt.quadpoints_phi
theta_study2d, phi_study2d = np.meshgrid(theta_study1d, phi_study1d)

# Plotting K
K_mag_study = np.sum(cp_opt.K()**2, axis=2)
plt.pcolor(theta_study1d, phi_study1d, K_mag_study, shading='nearest')
plt.colorbar()
plt.contour(theta_study2d, phi_study2d, K_mag_study)

# Contours for Phi
Phi_study = cp_opt.Phi()
plt.pcolor(theta_study1d, phi_study1d, Phi_study)
plt.colorbar()
plt.show()


# Contours for normal component
plt.pcolor(cp_opt.winding_surface.normal()[:,:,0])
plt.colorbar()
plt.show()
