# ====================================================================
#  Python-file
#     author:          William G. K. Martin
#     filename:        solver.py
# ====================================================================

"""
This modules contains functions for solving the ode assosiated with 
my three dimentional transport simulation.  The scheme assumes uniform
grid spacing and can be called for rays objects or by the matricies 
themselves. 

The solver requires a gridspacing, and the sigma, source, and incoming 
matricies of appropriate size.  The ode is solved along the first 
dimension of the arrays. 
"""


# import libraries
import scipy as sp
import scipy.integrate as int



# Solver for Rays class
def get_ray_rad(rays_obj, income=None):
    """ prepare get_solution to accept a rays_obj """
    r = rays_obj
    if income == None:
        return get_solution(r.rays_stride[0], 
                                 r['sigma'], 
                                 r['source'], sp.zeros(r.rays_shape[2:]))
    else:
        return get_solution(r.rays_stride[0], 
                                 r['sigma'], 
                                 r['source'], income)

# General solver for uniform grids
def get_solution(h, sig, emi, income, method='trap'):
    """ Use a stepping rule to get the values of the ode solution """
    # initialize solution
    u = sp.zeros(sig.shape)
    u[0] = income

    # compute stepping coeficients
    f = h * (emi[1::] + emi[0:-1:])
    c = 1.0 / (2.0 + h * sig[1::])
    step1 =  (2.0 - h * sig[0:-1:]) * c
    
    # step along the x-axis to solve the ode
    for i in range(f.shape[0]):
        u[i+1, ...] = (step1[i, ...] * u[i, ...] + 
                       c[i, ...] * f[i, ...])
    # return solution
    return u        



    # # initialize solution
    # u = sp.zeros(sig.shape)
    # u[0] = income

    # sd = 2 + h * sig[1::]
    # sl = -2 + h * sig[1:-1:]
    # alowertri = sp.diag(sd) + sp.diag(sl, k=-1)
    # ainv = sp.inv(alowertri)
    # f = h * (emi[1::] + emi[0:-1:])
    # f[0] = (2-h*sig[0])* u[0]
    
    
    


# def get_solution_trisolve(h, sig, emi, income, method='trap'):
#     """
#     Get the solution hopefully faster.
#     """
#     # initialize solution
#     u = sp.zeros(sig.shape)
#     u[0] = income

#     sd = 2 + h * sig[1::]
#     sl = -2 + h * sig[1:-1:]
#     alowertri = sp.diag(sd) + sp.diag(sl, k=-1)
#     ainv = sp.inv(alowertri)
#     f = h * (emi[1::] + emi[0:-1:])
#     f[0] = (2-h*sig[0])* u[0]
    
    
    


# f1 = get_ray_rad
# f2 = get_solution_trisolve
f1 = get_ray_rad
f2 = get_solution


class Solver:
    " Contrived class to bind methods to a class for testing "
    def get_ray_rad(self, *args):
        return f1(*args)
    def get_solution(self, *args):
        return f2(*args)

# Testing routine 
if __name__ == '__main__':
# -------------------------------------------------------------------------
# Define a dictionary to lookup function parameters for a few test cases 
# -------------------------------------------------------------------------
    fun_dict_lookup = {}  
    fun_dict_lookup['gauss_center_thin'] = {'fun_name':'gaussian', 
                                            'scale': 3,
                                            'args':(0.0, .005)}
    fun_dict_lookup['gauss_center_fat'] = {'fun_name':'gaussian',
                                           'scale': 3, 
                                           'args':(0.0, .1)}
    fun_dict_lookup['step_center_thin'] = {'fun_name':'step',
                                           'scale':1,
                                           'args':(0.0, .05)}
    fun_dict_lookup['one'] = {'fun_name':'const',
                              'scale':1,
                              'args':(0.0, 1.0)}
    fun_dict_lookup['zero'] = {'fun_name':'const',
                               'scale':0,
                               'args':(0.0, .5)}
    lambda_dict = {}
    lambda_dict['gaussian'] = lambda s, m, b: get_gauss_values(s, m, b)
    lambda_dict['step'] = lambda s, m, b: sp.where(abs(s-m)<=b, 1 / (2 * b), 0)
    lambda_dict['const'] = lambda s, m, b: sp.where(abs(s-m)<=100, 1, 0)
    
# -------------------------------------------------------------------------
    
# function for evaluating based on the dictionary above
    class MyFun(dict):
        """ define a object that looks up and returns functions """
        def __init__(self, name):
            """ define the name to be the name type and the get_values function
            to use the name to define the proper function """
            self.name = name
            self.b = 1.0
            
        def set_b(self, b):
            self.b = b
            
        def get_values(self, x):
            """ evaluate the appropriate function at the spcified locations 'x' """
            fun_dict = fun_dict_lookup[self.name]
            args = fun_dict['args']
            return fun_dict['scale'] * lambda_dict[fun_dict['fun_name']](x, *args)

        def od(self, a, b):
            integral = int.romberg(self.get_values, a, b)
            return sp.exp(-1.0 * integral)
        
        def od_to_b(self, a):
            return self.od(a, self.b)
        
        def compute_true(self, x, emi_fun, income):
            """ use high resolution quadrature to compute an accurate solution """
            u_true = sp.zeros(x.shape)
            
            # Loop over all the values of x
            for i, s in enumerate(x):
                self.set_b(s)
                source = lambda t: self.od_to_b(t) * emi_fun(t)
                u_true[i] = (income * self.od(-1.0, s) + int.romberg(source, -1.0, s))
            return u_true

### ====================================================================
#   Function library
### ====================================================================
    def get_gauss_values(s, m, b):
        """ define the values of a normal distribution, mean m, variance b."""
        if b > 0:
            c = 1 / sp.sqrt(2 * sp.pi * b)
            arg = - s**2.0 / (2 * b)
            return c * sp.exp(arg)
        else:
            raise er.InputError('Gaussian variancemust be positive!')

### ====================================================================
#   Test Solver Class using unittesting
### ====================================================================
if __name__ == '__main__':
    # import domain class to try solving some ode's 
    import sys
    # sys.path.append('/Users/will/myCode/transport3d_py/src/mixins')
    # sys.path.append('/Users/will/myCode/transport3d_py/src/classes')

    import enthought.mayavi.mlab as mlab
    import unittest
    import matplotlib.pyplot as plt

    # from rt3.src.classes
    import domain, options



    
    # setup spatial discritization
    N = 10
    fine = 40
    N_fine = (N - 1) * fine + 1
    x_fine = sp.linspace(-1, 1, N_fine)
    x = x_fine[0::fine]
    h = x_fine[1] - x_fine[0]            
    
    # define parameters
    sig_type, emi_type ='gauss_center_thin', 'gauss_center_fat'
    emi_myfun = MyFun(emi_type)
    emi_fun = emi_myfun.get_values
    emi = emi_fun(x_fine)
    sig_myfun = MyFun(sig_type)
    sig_fun = sig_myfun.get_values
    sig = sig_fun(x_fine)
    
    # setup solver and incoming values
    income = sp.array([1,])
    s = Solver()


    # use a step rule to solve
    u_fine = s.get_solution(h, sig, emi, income)
    u = u_fine[0::fine]
    u_true = sig_myfun.compute_true(x, emi_fun, income)
    rel_error = (u_true - u) / u_true.mean()


    plt.close(1)
    plt.figure(1)
    plt.plot(x_fine, u_fine, label='approx')
    plt.plot(x, u_true, label='true')
    plt.legend()
    plt.show()

    plt.close(2)
    plt.figure(2)
    plt.plot(x, rel_error, label='relative error desity')
    plt.legend()
    plt.show()

    # test solver
    class TestSolver(unittest.TestCase):
        
        def setUp(self):
            """ Prepare for each test. """
            self.s = Solver()
            self.N = 4
            self.M = 4

        def test_shape(self):
            """ Test that the input and output shapes are correct, using ParGrid arrays. """
            self.o = options.Options('o', self.N, self.M)
            self.g = domain.Grid('parg', self.o)
            h = self.g.grid_stride[0]
            sig = self.g['absorb']
            emi = self.g['emission']
            income = sp.ones(sig.shape[1::1])
            sol = self.s.get_solution(h, sig, emi, income)
            self.assert_(sol.shape == sig.shape)

        def test_gaussian_bumps(self):
            """ Test that gaussian bumps are integrated properly """
            # setup spatial discritization
            fine = 150
            N_fine = (self.N - 1) * fine + 1
            x_fine = sp.linspace(-1, 1, N_fine)
            x = x_fine[0::fine]
            h = x_fine[1] - x_fine[0]            
            
            # define parameters
            sig_type, emi_type ='gauss_center_thin', 'gauss_center_fat'
            emi_myfun = MyFun(emi_type)
            emi_fun = emi_myfun.get_values
            emi = emi_fun(x_fine)
            sig_myfun = MyFun(sig_type)
            sig_fun = sig_myfun.get_values
            sig = sig_fun(x_fine)
            
            # setup solver and incoming values
            s = Solver()
            income = sp.array([1,])

            # use a step rule to solve
            u_fine = s.get_solution(h, sig, emi, income)
            u = u_fine[0::fine]
            u_true = sig_myfun.compute_true(x, emi_fun, income)
            self.assert_((x_fine[0::fine] == x).all())
            self.assert_((abs(u_true - u)).sum() / u_true.sum() < 10.0**(-4))

    # run unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)


