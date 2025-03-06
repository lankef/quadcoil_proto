
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jax import jit, vmap, grad
import jax 
from jax.lax import while_loop, scan
from functools import partial
import numpy as np # DEBUG ONLY!!!
# import matplotlib.pyplot as plt

lstsq_vmap = vmap(jnp.linalg.lstsq)

def wl_debug(cond_fun, body_fun, init_val):
    val = init_val
    iter_num_wl = 1
    while cond_fun(val):
        val = body_fun(val)
    return val

def run_opt_optax(init_params, fun, max_iter, ftol, xtol, gtol, opt):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    # Carry is params, update, value, dx, du, df, state1
    def step(carry):
        params1, updates1, value1, _, _, _, state1 = carry
        value2, grad2 = value_and_grad_fun(params1, state=state1)
        updates2, state2 = opt.update(
            grad2, state1, params1, value=value2, grad=grad2, value_fn=fun
        )
        params2 = optax.apply_updates(params1, updates2)
        return(
            params2, updates2, value2, 
            jnp.linalg.norm(params2 - params1), 
            jnp.linalg.norm(updates2 - updates1), 
            jnp.abs(value2 - value1), 
            state2
        )
  
    def continuing_criterion(carry):
        params, _, _, dx, du, df, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | (
            (iter_num < max_iter) 
            & (err >= gtol)
            & (dx >= xtol)
            & (du >= xtol)
            & (df >= ftol)
        )
    init_carry = (
        init_params, 
        jnp.zeros_like(init_params, dtype=jnp.float32),
        0., 0., 0., 0.,
        opt.init(init_params)
    )
    final_params, final_updates, final_value, final_dx, final_du, final_df, final_state = while_loop(
        continuing_criterion, step, init_carry
    )
    return(otu.tree_get(final_state, 'params'))

# A scale factor making phi unit-free and order-1 is important
# because otherwise the gradient may be too shallow.
# We can't normalize A, b, c to ~1 but not do the same to Phi.
# current_scale is very necessary - if the unit for 
# phi is not removed, the gradients of this function 
# may be too flat or steep.
def eval_quad_scaled(phi_scaled, A, b, c, current_scale, ):
    # Evluates a quadratic function
    phi = phi_scaled/current_scale
    return((A@phi)@phi + b@phi + c)

# Must have signature 
# run_opt(x_km1, l_k, opt, max_iter_inner, tol_inner)

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
#     'run_opt',
#     'c_growth_rate',
#     'tol_outer',
#     'ftol_inner',
#     'xtol_inner',
#     'gtol_inner',
#     'max_iter_inner',
#     'max_iter_outer',
#     # 'scan_mode',
# ])
def solve_constrained(
        x_init,
        f_obj,
        run_opt,
        # No constraints by default
        c_init=0.1,
        lam_init=jnp.zeros(1, dtype=jnp.float32),
        h_eq=lambda x:jnp.zeros(1, dtype=jnp.float32),
        mu_init=jnp.zeros(1, dtype=jnp.float32),
        g_ineq=lambda x:jnp.zeros(1, dtype=jnp.float32),
        c_growth_rate=1.1,
        tol_outer=1e-5,
        ftol_inner=1e-5,
        xtol_inner=1e-5,
        gtol_inner=1e-5,
        max_iter_inner=500,
        max_iter_outer=20,
        # # Uses jax.lax.scan instead of while_loop.
        # # Enables history and forward diff but disables 
        # # convergence test.
        # scan_mode=False, 
    ):
    '''
    Solves 
    min f(x)
    subject to 
    h(x) = 0, g(x) <= 0
    '''

    # Has shape n_cons_ineq
    gplus = lambda x, mu, c: jnp.max(jnp.array([g_ineq(x), -mu/c]), axis=0)

    # True when non-convergent.
    @jit
    def outer_convergence_criterion(dict_in):
        # conv = dict_in['conv']
        x_k = dict_in['x_k']
        return(
            # This is the convergence condition (True when not converged yet)
            jnp.logical_and(
                dict_in['current_niter'] <= max_iter_outer,
                jnp.any(jnp.array([
                    # if gradient is too large, continue
                    jnp.any(g_ineq(x_k) >= tol_outer), 
                    # if equality constraints are too large or small, continue
                    jnp.any(h_eq(x_k) >= tol_outer),
                    jnp.any(h_eq(x_k) <= -tol_outer),
                    # if inequality constraints are too large, continue
                    jnp.any(g_ineq(x_k) >= tol_outer)
                ]))
            )
        )

    # Recursion
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
        ) 
        # Eq (10) on p160 of Constrained Optimization and Multiplier Method
        # Solving a stage of the problem
        x_k = run_opt(x_km1, l_k, max_iter_inner, ftol_inner, xtol_inner, gtol_inner)

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
        # if scan_mode:
        #     history_out = {
        #         'conv': jnp.linalg.norm(x_km1-x_k)/jnp.linalg.norm(x_k),
        #         'x_k': x_k,
        #         'objective': f_obj(x_k),
        #     }
        #     return(dict_out, history_out)
        return(dict_out)
    init_dict = body_fun_augmented_lagrangian({
        'conv': 100,
        'x_k': x_init,
        'c_k': c_init,
        'lam_k': lam_init,
        'mu_k': mu_init,
        'current_niter': 1,
    })
    # if scan_mode:
    #     result, history = scan(
    #         f=body_fun_augmented_lagrangian,
    #         init=init_dict,
    #         length=max_iter_outer
    #     )
    #     result['f_k'] = f_obj(result['x_k'])
    #     # Now we recover the l_k of the last augmented lagrangian iteration.
    #     c_k = result['c_k']
    #     lam_k = result['lam_k']
    #     mu_k = result['mu_k']
    #     l_k = lambda x: (
    #         f_obj(x) 
    #         + lam_k@h_eq(x) 
    #         + c_k/2 * (
    #             jnp.sum(h_eq(x)**2) 
    #             + jnp.sum(gplus(x, mu_k, c_k)**2)
    #         )
    #     ) 
    #     return(result, history, l_k)
    result = while_loop(
        cond_fun=outer_convergence_criterion,
        body_fun=body_fun_augmented_lagrangian,
        init_val=init_dict,
    )
    result['f_k'] = f_obj(result['x_k'])
    # Now we recover the l_k of the last augmented lagrangian iteration.
    c_k = result['c_k']
    lam_k = result['lam_k']
    mu_k = result['mu_k']
    l_k = lambda x: (
        f_obj(x) 
        + lam_k@h_eq(x) 
        + c_k/2 * (
            jnp.sum(h_eq(x)**2) 
            + jnp.sum(gplus(x, mu_k, c_k)**2)
        )
    ) 
    return(result)

# def solve_quad_constrained(
#         x_init,
#         c_init,
#         A_f, b_f, c_f,
#         run_opt, # 
#         current_scale=1,
#         # Equality constraints
#         lam_init=jnp.zeros(1, dtype=jnp.float32), # No constraints by default
#         A_eq=None, b_eq=None, c_eq=None,
#         # Inequality constraints
#         mu_init=jnp.zeros(1, dtype=jnp.float32), # No constraints by default
#         A_ineq=None, b_ineq=None, c_ineq=None,
#         # Parameters (static)
#         c_growth_rate=1.1,
#         tol_outer=1e-5,
#         ftol_inner=1e-5,
#         xtol_inner=1e-5,
#         gtol_inner=1e-5,
#         max_iter_inner=100,
#         max_iter_outer=15,
#         scan_mode=False,
# ):
#     f_obj = lambda x: eval_quad_scaled(x, A_f, b_f, c_f, current_scale)
#     if A_eq is None:
#         h_eq = lambda x:jnp.zeros(1, dtype=jnp.float32)
#     else:
#         # First normalize A, B, c by their size (the total constraint number).
#         h_eq = lambda x: eval_quad_scaled(x, A_eq, b_eq, c_eq, current_scale)
        
#     if A_ineq is None:
#         g_ineq=lambda x:jnp.zeros(1, dtype=jnp.float32)
#     else:
#         g_ineq=lambda x: eval_quad_scaled(x, A_ineq, b_ineq, c_ineq, current_scale)
    
# #     print('x_init', x_init.dtype)
# #     print('c_init', c_init)
# #     print('lam_init', lam_init.dtype)
# #     print('mu_init', mu_init.dtype)
# #     print('c_growth_rate', c_growth_rate)
#     return(
#         solve_constrained(
#             x_init,
#             c_init,
#             f_obj,
#             run_opt,
#             lam_init=lam_init,
#             mu_init=mu_init,
#             h_eq=h_eq,
#             g_ineq=g_ineq,
#             c_growth_rate=c_growth_rate,
#             tol_outer=tol_outer,
#             ftol_inner=ftol_inner,
#             xtol_inner=xtol_inner,
#             gtol_inner=gtol_inner,
#             max_iter_inner=max_iter_inner,
#             max_iter_outer=max_iter_outer,
#             # Uses jax.lax.scan instead of while_loop.
#             # Enables history and forward diff but disables 
#             # convergence test.
#             scan_mode=scan_mode,
#         )
#     )