# ========================================================================
#  Python-file
#     author:          William G. K. Martin
#     filename:        options.py
# ========================================================================

"""
Options for setting up the transport simulation.

"""


# Import 
#______________________________________________________________________
import sys
# sys.path.append('/Users/will/myCode/transport3d_py/src/classes')
# sys.path.append('/Users/will/myCode/transport3d_py/src/mixins')
# sys.path.append('/Users/will/myCode/transport3d_py/src/modules')

import scipy as sp
from scipy.linalg import norm

import parameter as param
# import rt3.src.modules.parameter as param


### ====================================================================
#   OPTIONS CLASS - specify simulation parameters 
### ====================================================================
class Options(object):
    """ This class should contain all information to setup a particular
    simulation """
    def __init__(self, name, N, M, group_name='group24'):
        """ instantiate an options class with the grid/direction 
        parameters N, and M. """
        self.name = name
        self.N = N
        self.M = M
        # group stuff
        self.group_name = group_name
        self.group_order = int(self.group_name[-2::])

        # grid stuff
        self.pos_shape = sp.array([3,] + [N,] * 3)
        self.grid_val_shape = self.pos_shape[1::]
        self.param_dict = {'absorb':'centerbump',
                           'scat':'centerbump',
                           'emission':'ybump'}
        self.grid_key_list = ['absorb', 'scat', 'emission', 
                              'ballistic', 'single', 'double', 'multiple',
                              'flux', 
                              'xflux', 'yflux', 'zflux'] # gridvalue keys
        # rays stuff               
        self.rays_shape = sp.array([3,] + [N,] * 3)
        self.rays_val_shape = self.rays_shape[1::]
        self.interp_method = 'linear'

        self.rays_key_list = ['emission', 'sigma', 
                             'rad'] # rayvalue keys
        # sphere stuff
        self.integrator_name = 'linear'
        self.fo_dir_shape = sp.array([3,] + [M, M])
        self.all_dir_shape = sp.array([3,] + [M * self.group_order, M])
        # incoming function type
        self.beam_type = ['doublehenyey',] *2 #* 8
        # position of incoming beam centers and henyey constants
        def normalize(direction):
            length = sp.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
            return direction/length       
        self.p = sp.array([normalize([1.0, 0.1, 0.0]),]*2)#,
#                            normalize([0.0, -1.0, 0.0]),
#                            normalize([-0.2, -0.1, -1.0])])## ,
#                            normalize([1, -.7, .7]),
#                            normalize([1, .7, -.7]),
#                            normalize([1, -.7, -.7]),
#                            normalize([-1, .7, -.7]),
#                            normalize([-1, -.7, .7]),
#                            normalize([-1, -.7, -.7])])

        self.pc = sp.array([.97,]*2)#, .97, .97])#, .90, .90, .90, .90, .90, .90])
        
        # direction of incoming henyey centers and henyey constants
        self.d = sp.array([normalize([-1, -0.35, -0.2]),]*2)#,
#                            normalize([0.0, 1.0, -.3]),
#                            normalize([0.3, 0.1, 1.0])])## ,
#                            normalize([-1.0, .3, -0.2]),
#                            normalize([-1.0, -.3, 0.2]),
#                            normalize([-1.0, .3, 0.2]),
#                            normalize([1.0, -.3, 0.2]),
#                            normalize([1.0, .3, -0.2]),
#                            normalize([1, 0.3, 0.2])])

        self.dc = sp.array([.99,]*2)#, .99, .99])#, .96, .96, .96, .96, .96, .96])

        
# convenience function
def normalize(direction):
    length = sp.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
            

### ====================================================================
#   OptionsUnitTest
### ====================================================================

if __name__ == '__main__':
    import unittest

    # Unittest for Options class
    class TestOptions(unittest.TestCase):
        
        def setUp(self):
            """ Prepare for each test. """
            self.N = 15         # grid resolution
            self.M = 15         # ray resolution
            self.options = Options('options', self.N, self.M, group_name='group24')
            # Define a list of attributes of options
            self.attribute_list = ['N', 'M', 
                                   'group_name', 'group_order', 
                                   'pos_shape', 'grid_val_shape', 
                                   'param_dict', 'grid_key_list',
                                   'rays_shape', 'rays_val_shape',
                                   'interp_method', 'rays_key_list',
                                   'beam_type', 'integrator_name',
                                   'fo_dir_shape', 'all_dir_shape']
            
            # Define a dictionary of shapes for all ndarray objects
            self.shape_dict = {
                'pos_shape': sp.array([3, self.N, self.N, self.N]),
                'grid_val_shape': sp.array([self.N, self.N, self.N]),
                'rays_shape': sp.array([3, self.N, self.N, self.N]),
                'rays_val_shape': sp.array([self.N, self.N, self.N]),
                'fo_dir_shape': sp.array([3, self.M, self.M]),
                'all_dir_shape': sp.array([3, self.M * self.options.group_order , self.M])}

            # Domain class member dictionary
            self.param_dict_key_list = ['absorb', 
                                        'scat', 
                                        'emission']   # grid parameters 
            self.dom_fun_list = param.fun_dict.keys() # functions on the ball
            self.interp_meth_list = ['trilin',
                                     'triquad']  # interpolation methods
            self.ode_meth_list = ['explicit', 
                                  'trap']        # solver methods
            self.grid_key_list = ['absorb', 'scat', 'emission',
                                  'ballistic', 'single', 
                                  'double', 'multiple',
                                  'xflux', 'yflux', 'zflux',
                                  'flux']        # gridvalue keys
            self.rays_key_list = ['emission', 'sigma', 
                                  'rad']         # rayvalue keys
                        
            # Sphere class member dictionary
            self.boundary_data_list = ['fan_beam']
            self.integrator_name_list = ['linear', 
                                         'spectral'] # integrators over S2
            
            # Group choice options
            self.group_name_list = ['group24', 
                                    'group48'] # groups of actions

        # test for existance and shape of attributes
        #____________________________________________________________________
        def test_options_attributes(self):
            """ test that all attributes in attribute list are defined """
            for a in self.attribute_list:
                self.assert_(hasattr(self.options, a), "attribute '"+a+"' not found")
        def test_shapes(self):
            """ Checking member shapes """
            for key, val in self.shape_dict.iteritems():
                c = 'testval = ' + 'self.options.' + key 
                exec c
                self.assert_((testval == val).all())
        def test_domain_options(self):
            """ Checking domain options """
            # check grid obj has required keys
            for v in self.grid_key_list:
                self.assert_(v in self.options.grid_key_list)
            # check that rays obj has required keys
            for v in self.rays_key_list:
                self.assert_(v in self.options.rays_key_list)
            # check all grid parameter functions are specified and supported 
            for fun in self.param_dict_key_list:
                self.assert_(self.options.param_dict.has_key(fun)) # all params specified
            for fun_type in self.options.param_dict.values():
                self.assert_(fun_type in self.dom_fun_list)
            
            
        def test_sphere_options(self):
            """ Checking sphere options """
            self.assert_(self.options.integrator_name in self.integrator_name_list)

        def test_group_options(self):
            """ Checking group options """
            self.assert_(self.options.group_name in self.group_name_list)


    # run unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptions)
    unittest.TextTestRunner(verbosity=2).run(suite)
