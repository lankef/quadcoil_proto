
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jax import jit, vmap
from jax.lax import while_loop, scan
from functools import partial

lstsq_vmap = vmap(jnp.linalg.lstsq)

def eval_quad_scaled(phi_scaled, A, b, c, current_scale, ):
    # Evluates a quadratic function
    phi = phi_scaled/current_scale
    return((A@phi)@phi + b@phi + c)


def run_opt(init_params, fun, opt, max_iter, tol):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
  
    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state
  
    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))
  
    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state

@jit
def solve_quad_unconstrained(A, b, c):
    x, _, _, _ = jnp.linalg.lstsq(2*A, -b)
    f = eval_quad_scaled(x, A, b, c, 1)
    return(x, f)

''' Constrained optimization '''


# A simple augmented Lagrangian implementation
# This jit flag is temporary, because we want 
# derivatives wrt f and g's contents too.
# @partial(jit, static_argnames=[
#     'f_obj',
#     'h_eq',
#     'g_ineq',
#     'opt',
#     'c_growth_rate',
#     'tol_outer',
#     'tol_inner',
#     'max_iter_inner',
#     'max_iter_outer',
#     'scan_mode',
# ])
def solve_constrained(
        x_init,
        c_init,
        f_obj,
        # No constraints by default
        lam_init=jnp.zeros(1),
        h_eq=lambda x:jnp.zeros(1),
        mu_init=jnp.zeros(1),
        g_ineq=lambda x:jnp.zeros(1),
        opt=optax.lbfgs(),
        c_growth_rate=1.1,
        tol_outer=1e-5,
        tol_inner=1e-5,
        max_iter_inner=500,
        max_iter_outer=20,
        # Uses jax.lax.scan instead of while_loop.
        # Enables history and forward diff but disables 
        # convergence test.
        scan_mode=False, 
    ):

    # Has shape n_cons_ineq
    gplus = lambda x, mu, c: jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)

    # if second_order_iter:
    #     gx_h_eq = jacobian(h_eq)

    # A simple convergence condition.
    # Tests whether the converegnce slows to a certain 
    # rate and the constraints are satisfied.
    @jit
    def convergence_criterion(dict_in):
        conv = dict_in['conv']
        x_k = dict_in['x_k']
        return(jnp.logical_not(
            # This is the convergence condition (True when converges)
            jnp.logical_or(
                jnp.all(jnp.array([
                    jnp.max(g_ineq(x_k)) <= tol_outer,
                    jnp.max(jnp.abs(h_eq(x_k))) <= tol_outer,
                    conv <= tol_outer
                ])), 
                dict_in['current_niter'] >= max_iter_outer
            )
        ))
        # return(jnp.logical_or(
        #     jnp.max(jnp.abs(opt_1)) >= tol_outer,
        #     jnp.max(jnp.abs(opt_2)) >= tol_outer
        # ))

    # Recursion
    # lam_k = lam_init
    # mu_k = mu_init
    # c_k = 10
    # x_km1 = phi_scaled_init
    @jit
    def body_fun_augmented_lagrangian(dict_in, x_dummy=None):
        x_km1 = dict_in['x_k']
        c_k = dict_in['c_k']
        lam_k = dict_in['lam_k']
        mu_k = dict_in['mu_k']
        l_k = lambda x: (
            f_obj(x) 
            + lam_k@h_eq(x) 
            + c_k/2 * (
                jnp.sum(h_eq(x)**2) 
                + jnp.sum(gplus(x, mu_k, c_k)**2)
            )
        ) # Eq (10) on p160 of Constrained Optimization and Multiplier Method
        # Solving a stage of the problem
        x_k, final_state = run_opt(x_km1, l_k, opt, max_iter_inner, tol_inner)

        lam_k_first_order = lam_k + c_k * h_eq(x_k)
        mu_k_first_order = mu_k + c_k * gplus(x_k, mu_k, c_k)

        dict_out = {
            'conv': jnp.linalg.norm(x_km1-x_k)/jnp.linalg.norm(x_k),
            'x_k': x_k,
            'c_k': c_k * c_growth_rate,
            'lam_k': lam_k_first_order,
            'mu_k': mu_k_first_order,
            'current_niter': dict_in['current_niter']+1,
        }
        # When using jax.lax.scan for outer iteration, 
        # the body fun also records history.
        if scan_mode:
            history_out = {
                'conv': jnp.linalg.norm(x_km1-x_k)/jnp.linalg.norm(x_k),
                'x_k': x_k,
                'objective': f_obj(x_k),
            }
            return(dict_out, history_out)
        return(dict_out)
    if scan_mode:
        result, history = scan(
            f=body_fun_augmented_lagrangian,
            init={
                'conv': 100,
                'x_k': x_init,
                'c_k': c_init,
                'lam_k': lam_init,
                'mu_k': mu_init,
                'current_niter': 1,
            },
            length=max_iter_outer
        )
        return(result, history)
    else:
        result = while_loop(
            cond_fun=convergence_criterion,
            body_fun=body_fun_augmented_lagrangian,
            init_val={
                'conv': 100,
                'x_k': x_init,
                'c_k': c_init,
                'lam_k': lam_init,
                'mu_k': mu_init,
                'current_niter': 1,
            },
        )
        return(result)

@partial(jit, static_argnames=[
    'opt',
    'c_growth_rate',
    'tol_outer',
    'tol_inner',
    'max_iter_inner',
    'max_iter_outer',
    'scan_mode',
])
def solve_quad_constrained(
        x_init,
        c_init,
        A_f, b_f, c_f,
        current_scale=1,
        # Equality constraints
        lam_init=jnp.zeros(1), # No constraints by default
        A_eq=None, b_eq=None, c_eq=None,
        # Inequality constraints
        mu_init=jnp.zeros(1), # No constraints by default
        A_ineq=None, b_ineq=None, c_ineq=None,
        # Parameters (static)
        opt=optax.lbfgs(),
        c_growth_rate=1.1,
        tol_outer=1e-5,
        tol_inner=1e-5,
        max_iter_inner=100,
        max_iter_outer=15,
        scan_mode=False,
):
    f_obj = lambda x: eval_quad_scaled(x, A_f, b_f, c_f, current_scale)
    if A_eq is None:
        h_eq = lambda x:jnp.zeros(1)
    else:
        h_eq = lambda x: eval_quad_scaled(x, A_eq, b_eq, c_eq, current_scale)
        
    if A_ineq is None:
        g_ineq=lambda x:jnp.zeros(1)
    else:
        g_ineq=lambda x: eval_quad_scaled(x, A_ineq, b_ineq, c_ineq, current_scale)
    
    return(
        solve_constrained(
            x_init,
            c_init,
            f_obj,
            lam_init=lam_init,
            mu_init=mu_init,
            h_eq=h_eq,
            g_ineq=g_ineq,
            opt=opt,
            c_growth_rate=c_growth_rate,
            tol_outer=tol_outer,
            tol_inner=tol_inner,
            max_iter_inner=max_iter_inner,
            max_iter_outer=max_iter_outer,
            # Uses jax.lax.scan instead of while_loop.
            # Enables history and forward diff but disables 
            # convergence test.
            scan_mode=scan_mode,
        )
    )