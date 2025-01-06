import unittest
import operator_helper
import f_b_and_k_operators_jax
import numpy as np
from simsopt import load
from simsopt.geo import SurfaceRZFourier
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve

class QuadcoilHelperTesting(unittest.TestCase):

    def test_grad(self):
        winding_surface, _ = load('surfaces.json')
        dg2 = winding_surface.gammadash2()
        dg1 = winding_surface.gammadash1()
        grad1, grad2 = operator_helper.grad_helper(
            gammadash1=winding_surface.gammadash1(), 
            gammadash2=winding_surface.gammadash2()
        )
        # Are the dot product identites satisfied?
        assert(np.all(np.isclose(np.sum(dg2 * grad2, axis=-1), 1)))
        assert(np.all(np.isclose(np.sum(dg1 * grad1, axis=-1), 1)))
        assert(np.all(np.isclose(np.sum(dg2 * grad1, axis=-1), 0)))
        assert(np.all(np.isclose(np.sum(dg1 * grad2, axis=-1), 0)))
        # Are the contravariant basis vectors perp to the unit normal?
        unitnormal = winding_surface.unitnormal()
        assert(np.all(np.isclose(np.sum(grad1 * unitnormal, axis=-1), 0)))
        assert(np.all(np.isclose(np.sum(grad2 * unitnormal, axis=-1), 0)))

    def test_norm(self):
        winding_surface, _ = load('surfaces.json')
        normal_vec = winding_surface.normal()
        (
            normN_prime_2d,
            inv_normN_prime_2d
        ) = operator_helper.norm_helper(normal_vec)
        assert(np.all(np.isclose(
            winding_surface.unitnormal()*normN_prime_2d[:,:,None],
            winding_surface.normal()
        )))
        assert(np.all(np.isclose(
            winding_surface.unitnormal(),
            winding_surface.normal()*inv_normN_prime_2d[:,:,None]
        )))
        assert(np.all(np.isclose(
            normal_vec,
            winding_surface.normal()
        )))

    def test_unitnormaldash(self):
        winding_surface, _ = load('surfaces.json')
        # Test in 10 random, small regions, because np.gradient
        # is only 2nd-order accurate and requires very fine grid
        # to agree with unitnormaldash.
        for i in range(10):
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
            theta_study1d, phi_study1d = winding_surface_hi_res.quadpoints_theta, winding_surface_hi_res.quadpoints_phi
            unitnormal = winding_surface_hi_res.unitnormal()
            unitnormaldash1, unitnormaldash2 = np.gradient(
                unitnormal, 
                phi_study1d,
                theta_study1d,
                axis=(0,1)
            )
            unitnormaldash1_test, unitnormaldash2_test = operator_helper.unitnormaldash(
                normal=winding_surface_hi_res.normal(),
                gammadash1=winding_surface_hi_res.gammadash1(),
                gammadash2=winding_surface_hi_res.gammadash2(),
                gammadash1dash1=winding_surface_hi_res.gammadash1dash1(),
                gammadash1dash2=winding_surface_hi_res.gammadash1dash2(),
                gammadash2dash2=winding_surface_hi_res.gammadash2dash2(),
            )
            # Remove the edge of both results because np.gradient
            # is inaccurate at the edges.
            unitnormaldash1_test = unitnormaldash1_test[1:-1,1:-1,:]
            unitnormaldash2_test = unitnormaldash2_test[1:-1,1:-1,:]
            unitnormaldash1 = unitnormaldash1[1:-1,1:-1,:]
            unitnormaldash2 = unitnormaldash2[1:-1,1:-1,:]
            print(
                'Test #', i+1, 
                'max error/max amplitude =', 
                np.max(unitnormaldash1_test-unitnormaldash1)/np.max(unitnormaldash1),
                np.max(unitnormaldash2_test-unitnormaldash2)/np.max(unitnormaldash2)
            )
            assert(np.all(np.isclose(
                unitnormaldash2_test, 
                unitnormaldash2
            )))
            assert(np.all(np.isclose(
                unitnormaldash1_test, 
                unitnormaldash1
            )))


    def test_K_dash(self):
        winding_surface, plasma_surface = load('surfaces.json')
        for i in range(10):
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
                194549.6432583 ,  1006169.72177152, -1677003.74430119,
                1750470.54137804,   471941.14387043, -1183493.44552104,
                1046707.62318593,  -334620.59690486,   658491.14959397,
                -1169799.54944824,  -724954.843765  ,  1143998.37816758,
                -2169655.54190455,  -106677.43308896,   761983.72021537,
                -986348.57384563,   532788.64040937,  -600463.7957275 ,
                1471477.22666607,  1009422.80860728, -2000273.40765417,
                2179458.3105468 ,   -55263.14222144,  -315581.96056445,
                587702.35409154,  -637943.82177418,   609495.69135857,
                -1050960.33686344,  -970819.1808181 ,  1467168.09965404,
                -198308.0580687 
            ]))
            cpst = CurrentPotentialSolve(cp_hi_res, plasma_surface, 0)
            cpst.B_matrix_and_rhs()   
            # The quadcoil vector. See paper.
            scaled_small_x = np.concatenate([cp_hi_res.get_dofs(), [1]])
            theta_study1d, phi_study1d = winding_surface_hi_res.quadpoints_theta, winding_surface_hi_res.quadpoints_phi
            K = cp_hi_res.K()
            # Calculating K's phi and theta derivatives with finite difference
            Kdash1_test, Kdash2_test = np.gradient(
                K, 
                phi_study1d,
                theta_study1d,
                axis=(0,1)
            )
            # Function to test
            (
                Kdash1_sv_op, 
                Kdash2_sv_op, 
                Kdash1_const,
                Kdash2_const
            ) = operator_helper.Kdash_helper(
                winding_surface_hi_res.normal(),
                winding_surface_hi_res.gammadash1(),
                winding_surface_hi_res.gammadash2(),
                winding_surface_hi_res.gammadash1dash1(),
                winding_surface_hi_res.gammadash1dash2(),
                winding_surface_hi_res.gammadash2dash2(),
                cp_hi_res.nfp, cp_hi_res.m, cp_hi_res.n,
                cp_hi_res.net_poloidal_current_amperes,
                cp_hi_res.net_toroidal_current_amperes,
                winding_surface_hi_res.quadpoints_phi,
                winding_surface_hi_res.quadpoints_theta,
                cp_hi_res.stellsym,
            )
            Kdash1_op = np.concatenate([
                Kdash1_sv_op,
                Kdash1_const[:, :, :, None]
            ], axis=-1)
            Kdash2_op = np.concatenate([
                Kdash2_sv_op,
                Kdash2_const[:, :, :, None]
            ], axis=-1)

            Kdash1 = Kdash1_op @ scaled_small_x
            Kdash2 = Kdash2_op @ scaled_small_x

            # Remove the edge of both results because np.gradient
            # is inaccurate at the edges.
            Kdash1_test = Kdash1_test[1:-1,1:-1,:]
            Kdash2_test = Kdash2_test[1:-1,1:-1,:]
            Kdash1 = Kdash1[1:-1,1:-1,:]
            Kdash2 = Kdash2[1:-1,1:-1,:]
            print(
                'Kdash test #', i+1, 
                'max error/max amplitude =', 
                np.max(Kdash1_test-Kdash1)/np.max(Kdash1),
                np.max(Kdash2_test-Kdash2)/np.max(Kdash2)
            )
            assert(np.all(np.isclose(
                Kdash2_test, 
                Kdash2
            )))
            assert(np.all(np.isclose(
                Kdash1_test, 
                Kdash1
            )))


if __name__ == "__main__":
    unittest.main()
