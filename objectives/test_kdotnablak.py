import unittest
import f_b_and_k_operators_jax
import grid_curvature_operator_jax
import numpy as np
from simsopt import load
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier
import sys
sys.path.insert(0, '..')
import utils

class QuadcoilKKTesting(unittest.TestCase):

    def test_KK(self):
        winding_surface, plasma_surface = load('surfaces.json')

        for i in range(5):
            # We compare the operator and a finite difference value of K dot grad K
            # in 10 small, random patches of the winding surface. This is because
            # np.gradient is only 2nd order accurate and needs very small grid size.
            loc1 = np.random.random()
            loc2 = np.random.random()

            winding_surface_hi_res = SurfaceRZFourier(
                    nfp=winding_surface.nfp, 
                    stellsym=winding_surface.stellsym, 
                    mpol=winding_surface.mpol, 
                    ntor=winding_surface.ntor, 
                    quadpoints_phi=np.linspace(loc1, loc1+0.0001, 200, endpoint=False), 
                    quadpoints_theta=np.linspace(loc2, loc2+0.0001, 200, endpoint=False), 
                )
            winding_surface_hi_res.set_dofs(winding_surface.get_dofs())
            cp_hi_res = CurrentPotentialFourier(
                winding_surface_hi_res, mpol=4, ntor=4,
                net_poloidal_current_amperes=11884578.094260072,
                net_toroidal_current_amperes=0,
                stellsym=True)
            cp_hi_res.set_dofs(np.array([  
                235217.63668779,  -700001.94517193,  1967024.36417348,
                -1454861.01406576, -1021274.81793687,  1657892.17597651,
                -784146.17389912,   136356.84602536,  -670034.60060171,
                194549.6432583,  1006169.72177152, -1677003.74430119,
                1750470.54137804,   471941.14387043, -1183493.44552104,
                1046707.62318593,  -334620.59690486,   658491.14959397,
                -1169799.54944824,  -724954.843765,  1143998.37816758,
                -2169655.54190455,  -106677.43308896,   761983.72021537,
                -986348.57384563,   532788.64040937,  -600463.7957275 ,
                1471477.22666607,  1009422.80860728, -2000273.40765417,
                2179458.3105468,   -55263.14222144,  -315581.96056445,
                587702.35409154,  -637943.82177418,   609495.69135857,
                -1050960.33686344,  -970819.1808181,  1467168.09965404,
                -198308.0580687 
            ]))
            cpst = CurrentPotentialSolve(cp_hi_res, plasma_surface, np.zeros(1024))
            cpst.B_matrix_and_rhs()   
            (_, _, _, _, current_scale) = f_b_and_k_operators_jax.f_B_and_current_scale(
                gj=cpst.gj,
                b_e=cpst.b_e,
                plasma_normal=plasma_surface.normal(),
                nfp=plasma_surface.nfp
            )
            phi = cp_hi_res.get_dofs()
            scaled_small_x = np.concatenate([current_scale * phi, [1]])
            scaled_x = scaled_small_x[:, None] * scaled_small_x[None, :]
            theta_study1d, phi_study1d = cp_hi_res.quadpoints_theta, cp_hi_res.quadpoints_phi
            G = cp_hi_res.net_poloidal_current_amperes
            I = cp_hi_res.net_toroidal_current_amperes
            normal_vec = cp_hi_res.winding_surface.normal()
            normN_prime_2d = np.sqrt(np.sum(normal_vec**2, axis=-1)) # |N|
            dK_dphi, dK_dtheta = np.gradient(
                cp_hi_res.K(), 
                phi_study1d,
                theta_study1d,
                axis=(0,1)
            )

            # Calculating the operator
            (
                K_dot_grad_K_operator_sv,
                K_dot_grad_K_linear,
                K_dot_grad_K_const,
            ) = grid_curvature_operator_jax.grid_curvature(
                # cp:CurrentPotentialFourier, 
                normal=winding_surface_hi_res.normal(),
                gammadash1=winding_surface_hi_res.gammadash1(),
                gammadash2=winding_surface_hi_res.gammadash2(),
                gammadash1dash1=winding_surface_hi_res.gammadash1dash1(),
                gammadash1dash2=winding_surface_hi_res.gammadash1dash2(),
                gammadash2dash2=winding_surface_hi_res.gammadash2dash2(),
                net_poloidal_current_amperes=cp_hi_res.net_poloidal_current_amperes,
                net_toroidal_current_amperes=cp_hi_res.net_toroidal_current_amperes,
                quadpoints_phi=winding_surface_hi_res.quadpoints_phi,
                quadpoints_theta=winding_surface_hi_res.quadpoints_theta,
                nfp=cp_hi_res.nfp, 
                cp_m=cp_hi_res.m, 
                cp_n=cp_hi_res.n,
                stellsym=cp_hi_res.stellsym,
            )
            (
                K_dot_grad_K_operator_sv_cyl,
                K_dot_grad_K_linear_cyl,
                K_dot_grad_K_const_cyl,
            ) = grid_curvature_operator_jax.grid_curvature_cylindrical(
                gamma=winding_surface_hi_res.gamma(),
                normal=winding_surface_hi_res.normal(),
                gammadash1=winding_surface_hi_res.gammadash1(),
                gammadash2=winding_surface_hi_res.gammadash2(),
                gammadash1dash1=winding_surface_hi_res.gammadash1dash1(),
                gammadash1dash2=winding_surface_hi_res.gammadash1dash2(),
                gammadash2dash2=winding_surface_hi_res.gammadash2dash2(),
                net_poloidal_current_amperes=cp_hi_res.net_poloidal_current_amperes,
                net_toroidal_current_amperes=cp_hi_res.net_toroidal_current_amperes,
                quadpoints_phi=winding_surface_hi_res.quadpoints_phi,
                quadpoints_theta=winding_surface_hi_res.quadpoints_theta,
                nfp=cp_hi_res.nfp, 
                cp_m=cp_hi_res.m, 
                cp_n=cp_hi_res.n,
                stellsym=cp_hi_res.stellsym,
            )

            K_dot_grad_K_test = (K_dot_grad_K_operator_sv@phi)@phi + K_dot_grad_K_linear@phi + K_dot_grad_K_const
            K_dot_grad_K_test_cyl = (K_dot_grad_K_operator_sv_cyl@phi)@phi + K_dot_grad_K_linear_cyl@phi + K_dot_grad_K_const_cyl

            K_dot_grad_K = (
                (cp_hi_res.Phidash1()[:, :, None]+G)*dK_dtheta
                -(cp_hi_res.Phidash2()[:, :, None]+I)*dK_dphi
            )/normN_prime_2d[:,:,None]

            K_dot_grad_K_cyl = utils.project_arr_cylindrical(gamma=cp_hi_res.winding_surface.gamma(), operator=K_dot_grad_K)

            # Remove the edge of both results because np.gradient
            # is inaccurate at the edges.
            K_dot_grad_K = K_dot_grad_K[1:-1,1:-1,:]
            K_dot_grad_K_test = K_dot_grad_K_test[1:-1,1:-1,:]
            K_dot_grad_K_cyl = K_dot_grad_K_cyl[1:-1,1:-1,:]
            K_dot_grad_K_test_cyl = K_dot_grad_K_test_cyl[1:-1,1:-1,:]

            print(
                'Test #', i, 
                'max error/max amplitude =', 
                np.max(K_dot_grad_K-K_dot_grad_K_test)/np.max(K_dot_grad_K)
            )
            print(
                'Test #', i, 
                'max error/max amplitude (cylindrical) =', 
                np.max(K_dot_grad_K_cyl-K_dot_grad_K_test_cyl)/np.max(K_dot_grad_K_cyl)
            )
            assert(np.all(np.isclose(
                K_dot_grad_K, 
                K_dot_grad_K_test
            )))

            assert(np.all(np.isclose(
                K_dot_grad_K_cyl, 
                K_dot_grad_K_test_cyl
            )))

if __name__ == "__main__":
    unittest.main()