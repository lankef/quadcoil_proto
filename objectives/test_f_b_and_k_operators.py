import unittest
import f_b_and_k_operators_jax
from operator_helper import A_b_c_to_block_operator
import jax.numpy as jnp
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
        # Loading testing example
        winding_surface, plasma_surface = load('surfaces.json')
        cp = CurrentPotentialFourier(
            winding_surface, mpol=4, ntor=4,
            net_poloidal_current_amperes=11884578.094260072,
            net_toroidal_current_amperes=0,
            stellsym=True)
        cp.set_dofs(jnp.array([  
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
        cpst = CurrentPotentialSolve(cp, plasma_surface, jnp.zeros(1024))
        # Pre-compute some important matrices
        cpst.B_matrix_and_rhs()   
        (A_f_B, b_f_B, c_f_B, B_normal, current_scale) = f_b_and_k_operators_jax.f_B_and_current_scale(
            gj=cpst.gj,
            b_e=cpst.b_e,
            plasma_normal=plasma_surface.normal(),
            nfp=cp.nfp
        )
        print('current_scale is', current_scale)

        # Test function for f_B
        def f_B_l2(Phi):
            A_times_phi = B_normal @ Phi
            f_B = 0.5*jnp.sum((A_times_phi - cpst.b_e) ** 2) * cp.nfp
            return(f_B)# Flattening K_dot_nabla_K and scaling

        phi = cp.get_dofs()
        scaled_small_x = jnp.concatenate([current_scale * phi, jnp.array([1])])
        scaled_x = scaled_small_x[:, None] * scaled_small_x[None, :]

        # Testing K
        A_K, b_K, c_K = f_b_and_k_operators_jax.K(
            normal=winding_surface.normal(),
            gammadash1=winding_surface.gammadash1(),
            gammadash2=winding_surface.gammadash2(),
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            quadpoints_phi=jnp.array(winding_surface.quadpoints_phi),
            quadpoints_theta=jnp.array(winding_surface.quadpoints_theta),
            nfp=winding_surface.nfp, 
            cp_m=cp.m, 
            cp_n=cp.n,
            stellsym=winding_surface.stellsym,
        )
        test_K_2 = (A_K@phi)@phi + b_K@phi + c_K
        test_K_1 = cp.K()
        assert(jnp.all(jnp.isclose(test_K_1, test_K_2)))

        # Testing K2
        A_K2, b_K2, c_K2 = f_b_and_k_operators_jax.K2(
            normal=winding_surface.normal(),
            gammadash1=winding_surface.gammadash1(),
            gammadash2=winding_surface.gammadash2(),
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            quadpoints_phi=winding_surface.quadpoints_phi,
            quadpoints_theta=winding_surface.quadpoints_theta,
            nfp=winding_surface.nfp, 
            cp_m=cp.m, 
            cp_n=cp.n,
            stellsym=winding_surface.stellsym,
        )
        test_K2_2 = (A_K2@phi)@phi + b_K2@phi + c_K2
        test_K2_1 = jnp.sum(test_K_2**2, axis=-1)[:A_K2.shape[0]]
        assert(jnp.all(jnp.isclose(test_K2_1, test_K2_2)))

        # Testing f_B
        (A_f_B, b_f_B, c_f_B, B_normal, current_scale) = f_b_and_k_operators_jax.f_B_and_current_scale(
            gj=cpst.gj,
            b_e=cpst.b_e,
            plasma_normal=plasma_surface.normal(),
            nfp=cp.nfp
        )
        assert(jnp.isclose(
            (A_f_B@phi)@phi + b_f_B@phi + c_f_B,
            f_B_l2(phi)
        ))

        # Testing block operators
        scaled_small_x = jnp.concatenate([current_scale * phi, jnp.array([1])])
        scaled_x = scaled_small_x[:, None] * scaled_small_x[None, :]
        K_block_operator, K_scale = A_b_c_to_block_operator(
            A=A_K, b=b_K, c=c_K, 
            current_scale=current_scale,
            normalize=True
        )
        K2_block_operator, K2_scale = A_b_c_to_block_operator(
            A=A_K2, b=b_K2, c=c_K2, 
            current_scale=current_scale,
            normalize=True
        )
        f_B_block_operator, f_B_scale = A_b_c_to_block_operator(
            A=A_f_B, b=b_f_B, c=c_f_B, 
            current_scale=current_scale,
            normalize=True
        )

        # Testing K
        test_K_3 = jnp.trace(K_block_operator@scaled_x, axis1=-1, axis2=-2) * K_scale
        assert(jnp.all(jnp.isclose(test_K_1, test_K_3)))

        # Testing K2
        test_K2_3 = jnp.trace(K2_block_operator@scaled_x, axis1=-1, axis2=-2) * K2_scale
        assert(jnp.all(jnp.isclose(test_K2_1, test_K2_3)))

        # Testing f_B
        assert(jnp.isclose(
            jnp.trace(f_B_block_operator@scaled_x, axis1=-1, axis2=-2) * f_B_scale,
            f_B_l2(phi)
        ))


if __name__ == "__main__":
    unittest.main()