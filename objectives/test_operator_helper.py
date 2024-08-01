import unittest
import operator_helper
import numpy as np
from simsopt import load
from simsopt.geo import SurfaceRZFourier

class QuadcoilHelperTesting(unittest.TestCase):

    def test_grad(self):
        winding_surface, _ = load('surfaces.json')
        dg2 = winding_surface.gammadash2()
        dg1 = winding_surface.gammadash1()
        grad1, grad2 = operator_helper.grad_helper(winding_surface)
        assert(np.all(np.isclose(np.sum(dg2 * grad2, axis=-1), 1)))
        assert(np.all(np.isclose(np.sum(dg1 * grad1, axis=-1), 1)))
        assert(np.all(np.isclose(np.sum(dg2 * grad1, axis=-1), 0)))
        assert(np.all(np.isclose(np.sum(dg1 * grad2, axis=-1), 0)))

    def test_norm(self):
        winding_surface, _ = load('surfaces.json')
        (
            normal_vec,
            normN_prime_2d,
            inv_normN_prime_2d
        ) = operator_helper.norm_helper(winding_surface)
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
            unitnormaldash1_test, unitnormaldash2_test = operator_helper.unitnormaldash(winding_surface_hi_res)
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

if __name__ == "__main__":
    unittest.main()
