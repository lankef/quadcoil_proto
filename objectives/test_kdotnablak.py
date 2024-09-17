import unittest
import f_b_and_k_operators
import grid_curvature_operator
import numpy as np
from simsopt import load
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier

class QuadcoilKKTesting(unittest.TestCase):

    def test_KK(self):
        winding_surface, plasma_surface = load('surfaces.json')

        for i in range(10):
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
            (
                _, 
                _, 
                current_scale, 
                _
            ) = f_b_and_k_operators.f_B_operator_and_current_scale(cpst)
            scaled_small_x = np.concatenate([current_scale * cp_hi_res.get_dofs(), [1]])
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
            K_dot_grad_K = (
                (cp_hi_res.Phidash1()[:, :, None]+G)*dK_dtheta
                -(cp_hi_res.Phidash2()[:, :, None]+I)*dK_dphi
            )/normN_prime_2d[:,:,None]
            # Calculating the operator
            K_dot_grad_K_operator = grid_curvature_operator.grid_curvature_operator(
                cp_hi_res, single_value_only=False, L2_unit=False, current_scale=current_scale
            )
            K_dot_grad_K_test = np.trace((K_dot_grad_K_operator @ scaled_x), axis1=-1, axis2=-2)
            # Remove the edge of both results because np.gradient
            # is inaccurate at the edges.
            K_dot_grad_K = K_dot_grad_K[1:-1,1:-1,:]
            K_dot_grad_K_test = K_dot_grad_K_test[1:-1,1:-1,:]

            print(
                'Test #', i, 
                'max error/max amplitude =', 
                np.max(K_dot_grad_K-K_dot_grad_K_test)/np.max(K_dot_grad_K)
            )
            assert(np.all(np.isclose(
                K_dot_grad_K, 
                K_dot_grad_K_test
            )))

if __name__ == "__main__":
    unittest.main()