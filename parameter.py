### ====================================================================
###  Python-file
###     author:          William G. K. Martin
###     filename:        parameter.py
### ====================================================================
#   responsible for difining parameter values using my library of standard 
#   funtions
### ====================================================================
import scipy as sp

#---------------------------------------------------------------------------------------
# list out the standard functions call
#---------------------------------------------------------------------------------------
fun_dict = {}
fun_dict['xbump'] = {'smooth':[True], 
                     'scale_factor':[1.0], 
                     'normalize':False, 
                     'centers':[sp.array([0.5, 0.0, 0.0])], 
                     'radii':[.3]}
fun_dict['ybump'] = {'smooth':[True], 
                     'scale_factor':[1.0], 
                     'normalize':False, 
                     'centers':[sp.array([0.0, 0.5, 0.0])], 
                     'radii':[.3]}
fun_dict['zbump'] = {'smooth':[True], 
                     'scale_factor':[1.0], 
                     'normalize':False, 
                     'centers':[sp.array([0.0, 0.0, 0.5])], 
                     'radii':[.3]}
fun_dict['centerbump'] = {'smooth':[True], 
                          'scale_factor':[1.0], 
                          'normalize':False, 
                          'centers':[sp.array([0.02, 0.01, 0.0])], 
                          'radii':[.4]}
fun_dict['asymetric'] = {'smooth':[True, False], 
                         'scale_factor':[3.0, .5], 
                         'normalize': False, 
                         'centers':[sp.array([0.0, 0.0, 0.0]), 
                                    sp.array([0.4, 0.2, 0.2])], 
                         'radii':[0.5, 0.2]}
fun_dict['constant'] = {'smooth':[False], 
                        'scale_factor':[3.0 / (4.0 * sp.pi)], 
                        'normalize': False, 
                        'centers':[sp.array([0.0, 0.0, 0.0])], 
                        'radii':[1]}
fun_dict['zero'] = {'smooth':[False], 
                    'scale_factor':[0.0], 
                    'normalize': False, 
                    'centers':[sp.array([0.0, 0.0, 0.0])], 
                    'radii':[1]}
a = .23
b = a * (1 + sp.sqrt(5)) / 2.0
fun_dict['icosahedron'] = {'smooth':[True] * 12, 
                           'scale_factor':[1] * 12, 
                           'normalize': False, 
                           'centers':[sp.array([0.0, a, b]), 
                                      sp.array([0.0, a, -b]), 
                                      sp.array([0.0, -a,  b]), 
                                      sp.array([0.0, -a, -b]), 
                                      sp.array([a, b, 0.0]), 
                                      sp.array([a, -b, 0.0]), 
                                      sp.array([-a,  b, 0.0]), 
                                      sp.array([-a, -b, 0.0]), 
                                      sp.array([b, 0.0, a]), 
                                      sp.array([-b, 0.0, a]), 
                                      sp.array([ b, 0.0, -a]), 
                                      sp.array([-b, 0.0, -a])], 
                           'radii':[0.18] * 12}
#---------------------------------------------------------------------------------------

def get_values(x, y, z, fun_type):
    """ Use the dictionary **fun_options to define values for smooth_bump functions 
    and step functions supported on the interior of small spheres """
    # Set the options for this function call based on the input fun_type
    smooth = fun_dict[fun_type]['smooth']
    centers = fun_dict[fun_type]['centers']
    radii = fun_dict[fun_type]['radii']
    normalize = fun_dict[fun_type]['normalize']
    scale_factor = fun_dict[fun_type]['scale_factor']
    # Initialize function values
    values = sp.zeros((x.shape[0], y.shape[1], z.shape[2]))
    # Loop over all bumps and steps to build values
    for i in range(len(smooth)):
        if smooth[i] is True:
            bump = smooth_bump(x, y, z, centers[i], radii[i])
            values = values + scale_factor[i] * bump
        else:
            values = values + scale_factor[i] * step(x, y, z, centers[i], radii[i])
    # Normalize all bumps by dividing by the total numver
    if normalize is True:
        values = values / len(smooth)
    return values

def step(x, y, z, center, radius):
    r_squared = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
    r_condition = radius**2
    height = 3.0 / (4.0 * sp.pi * radius**3)
    return sp.where(r_squared<r_condition, height, 0.0)

def smooth_bump(x, y, z, center, b):
    a = sp.sqrt(4.0 * sp.pi / (b**3 * (2.0 * sp.pi**2 - 15.0)))
    r = sp.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    cut = r < b
    bump = sp.zeros(r.shape)
    bump[r<b] = a * sp.cos(sp.pi / (2.0 * b) * r[r<b])**2.0 
    return bump

def norm(x, y, z):
    """ Define the Euclidian 2 norm """
    return sp.sqrt(x**2 + y**2 + z**2)

def normalize(direction):
    length = sp.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
    return direction/length

### ====================================================================
#   OptionsUnitTest
### ====================================================================

if __name__ == '__main__':
    import unittest
    # Unittest for Options class
    class TestOptions(unittest.TestCase):
        def setUp(self):
            """ Prepare for each test. """
            self.N = 15    
            self.test_shape = (self.N,) * 3
            self.pos = sp.random.random((3,) + self.test_shape)

        def test_signature(self):
            """ testing signature of 'get_values' """
            x = self.pos[0]
            y = self.pos[1]
            z = self.pos[2]
            # loop over all functions in dict
            for fun_type in fun_dict.keys():
                self.assertEqual(self.test_shape, 
                                 get_values(x, y, z, fun_type).shape)

    # run unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptions)
    unittest.TextTestRunner(verbosity=2).run(suite)
