from surfacerzfourier_jax import SurfaceRZFourierJAX
import jax.numpy as jnp

@tree_util.register_pytree_node_class
class QuadcoilProblem:
    def __init__(
        self, 
        winding_surface:SurfaceRZFourierJAX, # For evaluating integrals
        plasma_surface:SurfaceRZFourierJAX, # For evaluating integrals
        net_poloidal_current_amperes:float, net_toroidal_current_amperes:float,
        Bnormal_plasma:jnp.ndarray,
        nfp:int, stellsym:bool, mpol:int, ntor:int, 
        quadpoints_phi:jnp.ndarray, quadpoints_theta:jnp.ndarray # For evaluating objective
    ):
        # Defining surfaces
        self.winding_surface = winding_surface
        self.plasma_surface = plasma_surface
        
        # Defining currents and Bnormal_plasma
        self.net_poloidal_current_amperes = net_poloidal_current_amperes
        self.net_toroidal_current_amperes = net_toroidal_current_amperes
        self.Bnormal_plasma = Bnormal_plasma
        
        # Defining numerical parameters
        self.nfp = nfp
        self.stellsym = stellsym
        self.mpol = mpol
        self.ntor = ntor
        self.quadpoints_phi = quadpoints_phi
        self.quadpoints_theta = quadpoints_theta
        
    ''' Evaluation surface (copies WS with different quadpoints) '''
    def evaluation_surface(self):
        # Copies self.winding_surface with 
        # self.quadpoints_* instead. 
        # This way the winding surface used for 
        # integrals and the winding surface used
        # for evaluating quantities can have different resolutions.
        return SurfaceRZFourierJAX(
            nfp=self.winding_surface.nfp, 
            stellsym=self.winding_surface.stellsym, 
            mpol=self.winding_surface.mpol, 
            ntor=self.winding_surface.ntor, 
            quadpoints_phi=self.quadpoints_phi, 
            quadpoints_theta=self.quadpoints_theta, 
            dofs=self.winding_surface.get_dofs()
        )
    
        
    ''' JAX prereqs '''
    
    def tree_flatten(self):
        children = (
            self.winding_surface,
            self.plasma_surface,
            self.net_poloidal_current_amperes,
            self.net_toroidal_current_amperes,
            self.Bnormal_plasma,
            self.quadpoints_phi,
            self.quadpoints_theta,
        )
        aux_data = {
            'nfp': self.nfp,
            'stellsym': self.stellsym,
            'mpol': self.mpol,
            'ntor': self.ntor,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            winding_surface=children[0]
            plasma_surface=children[1]
            net_poloidal_current_amperes=children[2]
            net_toroidal_current_amperes=children[3]
            Bnormal_plasma=children[4]
            nfp=aux_data['nfp']
            stellsym=aux_data['stellsym']
            mpol=aux_data['mpol']
            ntor=aux_data['ntor']
            quadpoints_phi=children[5]
            quadpoints_theta=children[6]
        )