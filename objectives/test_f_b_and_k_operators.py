import unittest
import f_b_and_k_operators
import numpy as np
from simsopt import load
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve

class QuadcoilBKTesting(unittest.TestCase):

    """
    Testing the operators in f_b_and_k_operators. Thest includes:
    - f_B_operator_and_current_scale 
    The integrated normal field error f_B at the surface
    - K_operator_cylindrical
    The surface current K in a cylindrical coordinate
    - K_operator
    The surface current K in the xyz coordinate
    - K_theta
    The surface current K along the theta direction
    """

    def test_magnetic(self):
        # Loading testing example
        winding_surface, plasma_surface = load('surfaces.json')
        cp = CurrentPotentialFourier(
            winding_surface, mpol=4, ntor=4,
            net_poloidal_current_amperes=11884578.094260072,
            net_toroidal_current_amperes=0,
            stellsym=True)
        cp.set_dofs(np.array([  
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
        cpst = CurrentPotentialSolve(cp, plasma_surface, np.zeros(1024))
        # Pre-compute some important matrices
        cpst.B_matrix_and_rhs()
        (
            f_B_x_operator, 
            B_normal, 
            current_scale, 
            f_B_scale
        ) = f_b_and_k_operators.f_B_operator_and_current_scale(cpst)
        scaled_small_x = np.concatenate([current_scale * cp.get_dofs(), [1]])
        scaled_x = scaled_small_x[:, None] * scaled_small_x[None, :]

        ''' Testing the f_B operator '''
        def f_B_l2(Phi):
            A_times_phi = B_normal @ Phi
            f_B = 0.5*np.sum((A_times_phi - cpst.b_e) ** 2) * cp.nfp
            return(f_B)# Flattening K_dot_nabla_K and scaling
        assert(np.isclose(
            np.trace(f_B_x_operator @ scaled_x) * f_B_scale, 
            f_B_l2(cp.get_dofs())
        ))

        ''' Testing the K operator '''
        K_operator, K_scale = f_b_and_k_operators.K_operator(cp, current_scale)
        test_K_2 = np.trace((K_operator @ scaled_x) * K_scale, axis1=-1, axis2=-2)
        test_K_1 = cp.K()
        assert(np.all(np.isclose(test_K_1, test_K_2)))

        ''' Testing K l2 operator '''
        AK_l2_operator, AK_l2_scale = f_b_and_k_operators.K_l2_operator(cp, current_scale)
        test_K_l2_2 = np.trace((AK_l2_operator @ scaled_x)* AK_l2_scale, axis1=-1, axis2=-2)
        test_K_l2_1 = np.sum(test_K_2**2, axis=-1)
        assert(np.all(np.isclose(test_K_l2_1, test_K_l2_2)))


if __name__ == "__main__":
    unittest.main()