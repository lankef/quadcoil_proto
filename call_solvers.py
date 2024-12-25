import numpy as np
import scipy.sparse as sp


''' Recovering the  Phi vector from the X matrix (see paper) '''


def recover_phi(sol_x, current_scale):
    eigenvals, eigenvecs = np.linalg.eig(sol_x)
    max_eigen = np.argmax(eigenvals)
    quadcoil_x_scaled = eigenvecs[:, max_eigen] * np.sqrt(eigenvals[max_eigen])
    # Finding the second largest eigenvalue
    eigenvals[max_eigen] = 0
    second_max_eigenval = np.max(eigenvals)
    quadcoil_phi = quadcoil_x_scaled[:-1]/current_scale
    # Another method is simply taking the last row of X:
    # This doesn't minimize any norms, but the solution can
    # inform us about what sign to pick.
    quadcoil_phi_row = sol_x[-1, :-1]/current_scale
    if np.linalg.norm(quadcoil_phi + quadcoil_phi_row) < np.linalg.norm(quadcoil_phi - quadcoil_phi_row):
        quadcoil_phi = -quadcoil_phi
    return(quadcoil_phi, second_max_eigenval)


''' Matrix vectorization functions '''


def matrix_to_vec(A, solver):
    if solver=='sdpap':
        return(A.flatten())
    scale_arr = np.full_like(A, np.sqrt(2), dtype=np.float64)
    np.fill_diagonal(scale_arr, 1)
    # Get the lower triangular indices including the diagonal
    lower_tri_indices_with_diag = np.tril_indices(A.shape[0], k=0)
    if solver == 'scs':
        # This stacks the columns of the lower triangle portion of A
        return(np.flip(np.flip((A * scale_arr), axis=(0, 1))[lower_tri_indices_with_diag]))
    if solver == 'clarabel':
        # This stacks the columns of the lower triangle portion of A
        return((A * scale_arr)[lower_tri_indices_with_diag])

def vec_to_matrix(v, n, solver):
    if solver == 'scs':
        scale_arr = np.full((n, n), 1/np.sqrt(2), dtype=np.float64)
        np.fill_diagonal(scale_arr, 0.5)
        v_flip = np.flip(v)
        lower_tri_indices_with_diag = np.tril_indices(n, k=0)
        A_out = np.zeros((n, n))
        A_out[lower_tri_indices_with_diag] = v_flip
        A_out *= scale_arr
        A_out = np.flip(A_out, axis=(0,1))
        A_out += A_out.T
        return(A_out)
    if solver == 'clarabel':
        scale_arr = np.full((n, n), 1/np.sqrt(2), dtype=np.float64)
        lower_tri_indices_with_diag = np.tril_indices(n, k=0)
        np.fill_diagonal(scale_arr, 0.5)
        A_out = np.zeros((n, n))
        A_out[lower_tri_indices_with_diag] = v
        A_out *= scale_arr
        A_out += A_out.T
        return(A_out)
    if solver == 'sdpap':
        return(v.reshape((n, n)))

def matrix_to_vec_flat(A_flat, side_length, solver):
    # SCS wants the stacked column of the botton triangle matrix

    A = A_flat.reshape((side_length, side_length))
    return(matrix_to_vec(A, solver))

def stack_matrix_to_vec(A_stack, solver):
    side_length = A_stack.shape[-1]
    A_stack_flat = A_stack.reshape((A_stack.shape[0], -1))
    f_map = lambda A_flat: matrix_to_vec_flat(A_flat, side_length, solver)
    return(np.apply_along_axis(f_map, axis=1, arr=A_stack_flat))


''' Managing data matrices '''


def create_quadcoil_problem(F_matrix, solver):
    '''
    Creates a quadcoil problem in SCS from the operator F representing 
    f(Phi, a). The combined dimension of Phi, a will be inferred from the 
    shape of this operator. Therefore, F mist have shape 
    (n_dof + n_a + 1, n_dof + n_a + 1)
    '''
    if F_matrix.shape[0] != F_matrix.shape[1] or F_matrix.ndim != 2:
        raise ValueError('F_matrix is not a square matrix.')
    # The total number of dof is 
    # (n_dof + 1) * (n_dof + 2) / 2 + n_aux
    # len: (n_dof + 1) * (n_dof + 2) / 2
    F_vec = matrix_to_vec(F_matrix, solver=solver)
    n_dof_vec_x = len(F_vec)
    # Matrix for enforcing X[n_dof+1, n_dof+1] = 1
    # Create a sparse matrix with one column (n x 1)
    A_last_element = sp.csc_matrix(
        (np.array([1]) , (np.array([0]), np.array([n_dof_vec_x-1]))),
        shape=(1, n_dof_vec_x)
    )
    A_sdp = -sp.identity(n_dof_vec_x, format='csc')
    
    # SCS, clarabel and SDPA all solve
    # min x^T P x + c^T x
    # subject to 
    # Ax + s \geq b
    # where s are different types of cones.
    # The A arrays must have shape(n_cons, n_dof_tot)
    data_dict = {
        # The c vector
        'c': F_vec,
        # The size of the positive semidefinite cone
        's_positive_semidefinite': F_matrix.shape[0],
        # The part of A enforcing the positive semidefinite constraint
        'A_X_positive_semidefinite': A_sdp,
        'b_X_positive_semidefinite': np.zeros(A_sdp.shape[0]),
        # The A and b enforcing constraints.
        # Rows of A corresponds to constraint index.
        # Cols of A corresponds to var index.
        'A_positive': sp.csc_matrix((0, n_dof_vec_x)),
        # A blank 1d array
        'b_positive': np.zeros((0)), 
        'A_zero': A_last_element,
        'b_zero': np.array([1.]),
        # The number of auxillary variables
        'num_aux_ineq': 0,
        'num_aux_eq': 0,
    }
    return(data_dict)

def add_constraints_and_aux(data_dict, solver, G_j, c=None, d_j=None, e_j=0, ineq_mode=True):
    '''
    Add a set of constraints and associated auxillary variables. 
    QUADCOIL, at the moment, does not support adding constraint sets using 
    existing d_j. To do that, stack the G_j, c_j, d_j and e_j of 
    constraints sharing one set of components in d.
    (new inequality constraint!)

    When ineq_mode is True, adds a set of constraints of form:
    <G_j, X> + d_j * b \leq e_j

    When ineq_mode is False, adds a set of constraints of form:
    <G_j, X> + d_j * b = e_j

    G_j is a stack of operators G_j: [n_cons, n_dof+1, n_dof+1]
    c_j and d_j are 0 by default. This adds no additional auxillary variable.
    c has shape [n_dof_aux]
    d_j has shape [n_cons, n_dof_aux]
    e_j has shape [n_cons]
    '''
    # A stack of vectorized matrices
    # shape: (n_grid or n_grid/2, (n_dof+1) x (n_dof+2)/2)
    A_x = sp.csc_matrix(stack_matrix_to_vec(G_j, solver=solver))
    n_cons = G_j.shape[0]

    if np.isscalar(e_j):
        e_j = np.full(n_cons, e_j)

    if ineq_mode:
        # Corresponds to the positive cone in SCS
        mode = 'positive'
        mode_other = 'zero'
    else:
        # Corresponds to the zero cone in SCS
        mode = 'zero'
        mode_other = 'positive'
    # Loading the existing data matrix
    A_original = data_dict['A_' + mode]
    # First, padding A_x on the right to match the column count of A_original.
    padding_A_x = sp.csc_matrix((A_x.shape[0], A_original.shape[1] - A_x.shape[1]))
    A_x_padded = sp.hstack([A_x, padding_A_x])
    # Stacking new constraints on Phi and a to new rows at the bottom left 
    # corner of the data matrix
    A_new_rows = sp.vstack([A_original, A_x_padded], format='csc')
    if d_j is None or c is None:
        if (d_j is not None) or (c is not None):
            raise AttributeError('c and d_j must both be None or not None.')
        # Adding new rows to the data matrix (new constraints)
        # but no new columns (new vars)
        data_dict['A_' + mode] = A_new_rows
    # Only runs when this constraint introduces new auxillary variables
    else:
        n_dof_aux = len(c)
        if d_j.shape != (n_cons, n_dof_aux):
            raise AttributeError('Shape mismatch in c and d_j')

        # Create zero matrices for padding the upper right block of the new 
        # data matrix. 
        zero_top_right = sp.csc_matrix((A_original.shape[0], n_dof_aux))

        # Create new columns in the data matrix to accomodate the newly added
        # auxillary variables
        A_new_cols = sp.vstack([zero_top_right, d_j], format='csc')

        # Add both new rows and cols to the data matrix
        data_dict['A_' + mode] = sp.hstack([A_new_rows, A_new_cols], format='csc')

        # Add blank cols to the other data matrices
        A_other = data_dict['A_' + mode_other]
        A_other_new_cols = sp.csc_matrix((A_other.shape[0], n_dof_aux))
        data_dict['A_' + mode_other] = sp.hstack([A_other, A_other_new_cols], format='csc')

        A_sd = data_dict['A_X_positive_semidefinite']
        A_sd_new_cols = sp.csc_matrix((A_sd.shape[0], n_dof_aux))
        data_dict['A_X_positive_semidefinite'] = sp.hstack([A_sd, A_sd_new_cols], format='csc')

        # Appending to the linear objective vector
        data_dict['c'] = np.concatenate((data_dict['c'], c))

        # Appending zeros to the other matrices

    # Appending to the data vector
    data_dict['b_' + mode] = np.concatenate((data_dict['b_' + mode], e_j))


''' Converting common objective functions into the QUADCOIL format '''


def add_max_abs_constraint(data_dict, solver, operator, cons):
    if operator.ndim != 3:
        raise AttributeError('Operator must have 3 dimensions.')
    if operator.shape[1] != operator.shape[2]:
        raise AttributeError('Operator\'s last two dimensions must be the same')
    # The linf norm problem can be written as 4 types of constraints:
    # -tr(AKK_operator[i, :, :] @ cvxpy_X) <= cons
    # G_j = -operator
    #  tr(AKK_operator[i, :, :] @ cvxpy_X) <= cons
    # G_j =  operator
    stack_height = operator.shape[0]
    G_stacked = np.concatenate((
        -operator,
        operator,
    ), axis=0)
    e_stacked = np.concatenate((
        np.full(2 * stack_height, cons),
    ))
    add_constraints_and_aux(
        data_dict, 
        solver, 
        G_j=G_stacked, 
        e_j=e_stacked,
    )


# Prototype of Linf norm constraint. This is unnecessarily complex, but 
# something similar will be needed when we are minimizing an L-inf norm instead
# of constraining it.
# Nevertheless, can be used to test auxillary variable handling.
def add_max_abs_constraint_aux(data_dict, solver, operator, cons):
    if operator.ndim != 3:
        raise AttributeError('Operator must have 3 dimensions.')
    if operator.shape[1] != operator.shape[2]:
        raise AttributeError('Operator\'s last two dimensions must be the same')
    # The linf norm problem can be written as 4 types of constraints:
    # -tr(AKK_operator[i, :, :] @ cvxpy_X) - p <= 0
    # G_j = -operator
    # d_j =  stack of [-1]
    #  tr(AKK_operator[i, :, :] @ cvxpy_X) - p <= 0
    # G_j =  operator
    # d_j =  stack of [-1]
    #                                      + p <= cons
    # G_j =  square of [0]
    # d_j =  [+1]
    #                                      - p <= 0
    # G_j =  square of [0]
    # d_j =  [-1]
    stack_height = operator.shape[0]
    stack_matrix_size = operator.shape[1]
    d_stacked = np.concatenate((
        -np.ones(2 * stack_height), 
        np.array([1, -1])
    ))[:, None]
    G_stacked = np.concatenate((
        -operator,
        operator,
        np.zeros((2, stack_matrix_size, stack_matrix_size))
    ), axis=0)
    e_stacked = np.concatenate((
        np.zeros(2 * stack_height),
        np.array([cons, 0])
    ))
    add_constraints_and_aux(
        data_dict, 
        solver, 
        G_j=G_stacked, 
        d_j=d_stacked, 
        e_j=e_stacked,
        c=np.zeros(1)
    )


try:
    import scs
    def scs_solve(data_dict, current_scale, warm_start_args={}, **kwargs):
        '''
        Generates a SCS problem from data.
        '''
        cone = {
            'z': data_dict['A_zero'].shape[0], # Size of the zero cone
            'l': data_dict['A_positive'].shape[0], # Size of the positive cone
            's': data_dict['s_positive_semidefinite'], # Size of the positive semidefinite cone
        }
        data = {
            'P': None,
            'A': sp.vstack((
                data_dict['A_zero'],
                data_dict['A_positive'],
                data_dict['A_X_positive_semidefinite'],
            )),
            'b': np.concatenate((
                data_dict['b_zero'],
                data_dict['b_positive'],
                data_dict['b_X_positive_semidefinite'],
            )),
            'c': data_dict['c']
        }
        solver = scs.SCS(data, cone, eps_abs=1e-6, eps_rel=1e-6, **kwargs)
        sol = solver.solve(**warm_start_args)
        sol_x = vec_to_matrix(sol['x'], data_dict['s_positive_semidefinite'], solver='scs')
        # One method to extract (Phi, a) from its self-outer-product
        # is using X's eigen-decomposition. The result from this method
        # has ambiguous signs.
        # This is also the Frobenius norm minimizing, rank-1 approximation 
        # of X.
        size_phi = data_dict['s_positive_semidefinite'] * (data_dict['s_positive_semidefinite'] + 1) // 2
        quadcoil_phi, second_max_eigenval = recover_phi(sol_x[:size_phi], current_scale)
        print('Solve time:', sol['info']['solve_time'], 'ms')
        print('The second largest eigenvalue is', second_max_eigenval)
        return(quadcoil_phi, second_max_eigenval, sol_x[size_phi:], solver, sol)
except: 
    print('The solver SCS is unavailable.')

try:
    import clarabel
    def clarabel_solve(data_dict, current_scale, **kwargs):
        '''
        Generates a clarabel problem from data.
        '''
        cones_list = [
            clarabel.ZeroConeT(data_dict['A_zero'].shape[0]),
            clarabel.NonnegativeConeT(data_dict['A_positive'].shape[0]),
            clarabel.PSDTriangleConeT(data_dict['s_positive_semidefinite']),
        ] 

        A = sp.vstack((
            data_dict['A_zero'],
            data_dict['A_positive'],
            data_dict['A_X_positive_semidefinite'],
        ))
        b = np.concatenate((
            data_dict['b_zero'],
            data_dict['b_positive'],  
            data_dict['b_X_positive_semidefinite'],
        ))
        q = data_dict['c']
        P_size = data_dict['A_zero'].shape[-1]
        P = sp.csc_matrix((P_size, P_size))

        settings = clarabel.DefaultSettings()
        # settings.direct_solve_method = 'mkl'
        # settings.verbose = False
        solver = clarabel.DefaultSolver(P, q, A, b, cones_list, settings)
        solution = solver.solve()
        size_phi = data_dict['s_positive_semidefinite'] * (data_dict['s_positive_semidefinite'] + 1) // 2
        quadcoil_phi, second_max_eigenval = recover_phi(
            vec_to_matrix(
                solution.x[:size_phi], 
                data_dict['s_positive_semidefinite'], 
                solver='clarabel'
            ),
            current_scale
        )
        print('The second largest eigenvalue is', second_max_eigenval)
        return(quadcoil_phi, second_max_eigenval, solution.x[size_phi:], solver, solution)
except: 
    print('The solver Clarabel is unavailable.')

try:
    import sdpap
    def sdpap_solve(data_dict, current_scale, **kwargs):
        '''
        Generates a clarabel problem from data.
        '''
        print('Warning: The auxillary variables is not be working correctly yet.')
        print('Warning: sdpa cannot solve the no-windowpane problem well yet.')
        K = sdpap.SymCone(
            l=len(data_dict['c']) - data_dict['s_positive_semidefinite'],
            s=(data_dict['s_positive_semidefinite'],) 
        )
        print('s_positive_semidefinite', (data_dict['s_positive_semidefinite'],))
        J = sdpap.SymCone(
            f=data_dict['A_zero'].shape[0], # Size of the zero cone
            l=data_dict['A_positive'].shape[0], # Size of the positive cone
        )
        A = sp.vstack((
            data_dict['A_zero'],
            data_dict['A_positive'],
        ))
        print('A_zero', data_dict['A_zero'].shape)
        print('A_positive', data_dict['A_positive'].shape)
        print('A_X_positive_semidefinite', data_dict['A_X_positive_semidefinite'].shape)
        print('b_zero', data_dict['b_zero'].shape)
        print('b_positive', data_dict['b_positive'].shape)
        print('b_X_positive_semidefinite', data_dict['b_X_positive_semidefinite'].shape)
        b = np.concatenate((
            data_dict['b_zero'],
            data_dict['b_positive'],  
        ))
        print(b.shape)
        c = data_dict['c']

        x, y, sdpapinfo , timeinfo , sdpainfo = sdpap.solve(A,b,c,K,J)
        size_phi = data_dict['s_positive_semidefinite']**2
        print('x[:size_phi]', type(x[:size_phi, 0]))
        quadcoil_phi, second_max_eigenval = recover_phi(
            vec_to_matrix(
                x.toarray()[:size_phi, 0], 
                data_dict['s_positive_semidefinite'],
                solver='sdpap'
            ),
            current_scale
        )
        print('The second largest eigenvalue is', second_max_eigenval)
        return(quadcoil_phi, second_max_eigenval, x[size_phi:], (sdpapinfo, timeinfo, sdpainfo), {'x': x, 'y': y})
except:
    print('The solver sdpa-python (sdpap) is unavailable.')

def solve_quadcoil(data_dict, current_scale, solver='clarabel', **kwargs):
    # Testing shapes
    assert(data_dict['A_positive'].shape[1] == data_dict['A_zero'].shape[1])
    assert(data_dict['A_positive'].shape[1] == data_dict['c'].shape[0])
    print('Solving QUADCOIL. solver =', solver)
    print('The problem consists of:')
    print('# unknowns:                                ', data_dict['c'].shape[0])
    print('# auxillary among unknowns:                ', len(data_dict['c']) - data_dict['s_positive_semidefinite'])
    print('# equality constraints:                    ', data_dict['A_zero'].shape[0]-1, '+ 1 from Shor relaxation')
    print('# inequality constraints:                  ', data_dict['A_positive'].shape[0])
    print('The positive semi-definite matrix has size:', data_dict['s_positive_semidefinite'])
    if solver == 'scs':
        return(scs_solve(data_dict, current_scale, **kwargs))
    if solver == 'clarabel':
        return(clarabel_solve(data_dict, current_scale, **kwargs))
    if solver == 'sdpap':
        return(sdpap_solve(data_dict, current_scale, **kwargs))