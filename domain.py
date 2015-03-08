# ====================================================================
#  Python-file
#     author:          William G. K. Martin
#     filename:        domain.py
# ====================================================================

"""
This domain class is a dictionary containing function values on a 
three-dimentional uniform grid.  Since the goal is to solve transport 
the dictionary will contain optical parameters of the PDE and the 
flux-density obtained by solving the equation.  

Two types of domains are important enough to have their own subclass.
The first domain subclass is called "Grid" and it is responsible both
for organizing parameter data before the main computation and for
storing and visualizing solution data after the main computation.  
The computations for local subsets of directions are performed by
an object called "Rays".  Rays is responsible for using low level 
interpolators and ode-solvers generate radience values allong rays.

Grid handels global organization and data storage, while Rays handles 
the interpolating and ode-solving cores.  
"""

# Import libraries
#__________________________________________________________________
import copy as copy

import scipy as sp
import enthought.mayavi.mlab as mlab

from my_plots import plot_grid_list
import group, options, domain_mixins, solver, parameter, interp
opt, mix, param = options, domain_mixins, parameter


# from rt3.src.modules.my_plots import plot_grid_list
# from rt3.src.classes import group, options
# opt = options
# from rt3.src.mixins import domain_mixins 
# mix = domain_mixins
# from rt3.src.modules import solver, parameter, interp
# param = parameter



#import solver
# import options as opt
#import parameter as param

#import interp as interp

#   Function definitions 
#__________________________________________________________________


# group action generator dictionary (for moving points)
act_lambda = {'' : lambda x: x,
              'a' : lambda x: x[::-1, :, :],
              'b' : lambda x: x[:, ::-1, :],
              'c' : lambda x: x[:, :, ::-1],
              'd' : lambda x: x.transpose([0, 2, 1]),
              'e' : lambda x: x.transpose([2, 1, 0]),
              'f' : lambda x: x.transpose([1, 0, 2]),
              's' : lambda x: x.transpose([2, 0, 1])}


#   Class definitions 
#__________________________________________________________________

class Domain(type({})):
    """ meta-class parent to Grid and Rays. """
    def __init__(self, name, opt):
        """ Define pos and support mask """
        # define members of domain
        self.name = name
        self.scat_order = 0
        self.N = opt.N
        self.grid_shape = opt.grid_val_shape
        self.rays_shape = opt.rays_val_shape
        self.grid_stride = (2.0 / (self.grid_shape[0]-1.0), 
                            2.0 / (self.grid_shape[1]-1.0), 2.0 
                            / (self.grid_shape[2]-1.0))
        self.rays_stride = (2.0 / (self.rays_shape[0]-1.0), 
                           2.0 / (self.rays_shape[1]-1.0), 
                           2.0 / (self.rays_shape[2]-1.0))
        h = self.grid_stride
        x, y, z = sp.ogrid[-1.0:1.0+h[0]/2:h[0], -1.0:1.0+h[1]/2:h[1], -1.0:1.0+h[2]/2:h[2]]
        self.x, self.y, self.z = sp.broadcast_arrays(x, y, z)
        self.radius = sp.sqrt(x**2 + y**2 + z**2)
        self.supp_rad = 1 - 2.5 * h[0] / 4.0
        self.supp = abs(self.radius) < self.supp_rad
        self.pos = sp.array([self.x, self.y, self.z])


# Subclass Domain to define Grid
#__________________________________________________________________

class Grid(Domain):        
    """ 
    Subclass Domain to add action, parameter definition, solution 
    storage, and SOS convergence testing methods.         

    Grid is stores flux values and un altered function values 
    in the dictionary (self), and stores matrix values that are 
    frequently reassigned as attribures.
    """
    def __init__(self, name, opt):
        """ 
        Initialize data structure.
        """
        # inherit domain stuff
        super(Grid, self).__init__(name, opt)
        self.group = group.Group(opt)

        # initialize rays and function values as zero
        for key in opt.grid_key_list:
            self[key] = sp.zeros(opt.grid_val_shape)

        # define parameters with param.get_values
        self.param_dict = opt.param_dict
        for key, val in self.param_dict.iteritems():
            self[key] = sp.where(self.supp, 
                                 param.get_values(self.x, 
                                                  self.y, 
                                                  self.z,
                                                  val), 0.0)

        # define data attr and instantiation values 
        self.sigma = self['absorb'] + self['scat']
        self.source = sp.zeros(opt.grid_val_shape)
        self.loop_flux = sp.zeros(opt.grid_val_shape)
        self.loop_vflux = sp.zeros((3,) + tuple(opt.grid_val_shape))
        self.act = ''
        self.ds = None
        self.dv = None
        self.scat_order = 0


    # methods for setting members
    #__________________________________________________________________

    def setup_scatter_loop(self):
        """ define source function for loop and set loop_flux's to zero """
        # define source based on scattering order
        if self.scat_order == 0:
            self.source = self['emission'].copy()
        if self.scat_order >= 1:
            self.source = self.loop_flux * self['scat']
#### check plot ####
#        plot_grid_list(self.x, self.y, self.z, [self.loop_flux, self['scat']], ['loop_flux', 'scat'] )
#### check plot ####



        # reset loop_fluxes to zero
        self.loop_flux[...] = 0.0
        self.loop_vflux[...] = 0.0


    def set_new_ds(self, ds):
        """ Set ds for a new loop over all group actions """
        # set current ds
        self.ds = ds

        
    def set_new_act(self, act):
        """ rotate data by act """
        # update the current acted_on state of grid
        self.act = self.group.compose(act, self.act)
        self.inv = self.group.inverse(act)
        if self.group.compose(self.inv, self.act) != '':
            raise valueError('group.act, group.ive error')

        # define the vector surface element
        self.dv = self.ds[self.act] * self.ds.weight

        # left act on functions 
        act = list(act)[:]         # string to list for popping
        while act:
            gen = act.pop()
            self.sigma = act_lambda[gen](self.sigma)
            self.source = act_lambda[gen](self.source)
#             self.sigma[...] = act_lambda[gen](self.sigma)
#             self.source[...] = act_lambda[gen](self.source)


    def update_loop_fluxes(self, grid_rad):
        """ store data and check for convergence """
        # Rotate arrays back before storage
        inv = list(self.inv)[:]         # string to list for popping
        while inv:
            gen = inv.pop()
            self.sigma = act_lambda[gen](self.sigma)
            self.source = act_lambda[gen](self.source)
            grid_rad = act_lambda[gen](grid_rad)
        self.act = self.group.compose(self.inv, self.act)

        # update loop fluxes
        self.loop_flux += self.ds.weight * grid_rad
        self.loop_vflux += sp.outer(self.dv, grid_rad).reshape((3,) + grid_rad.shape)

        
    def store_loop_fluxes(self):
        """ 
        use the loop fluxes that have been computed throughout this loop
        to update the final fluxes stored in the grid dictionary. 
        """
        # update scalar flux values
        self['flux'] += self.loop_flux
        if self.scat_order == 0:
            self['ballistic'][...] = self.loop_flux
        elif self.scat_order == 1:
            self['single'][...] = self.loop_flux
        elif self.scat_order == 2:
            self['double'][...] = self.loop_flux
        elif self.scat_order >= 3:
            self['multiple'] += self.loop_flux



        # update scattering order
        self.scat_order += 1

        # update vector flux values
        self['xflux'] += self.loop_vflux[0]
        self['yflux'] += self.loop_vflux[1]
        self['zflux'] += self.loop_vflux[2]

        # plot_grid_list(self.x, self.y, self.z, [self.loop_flux, self.loop_vflux], ['loop_flux', 'loop vflux'] )






    # functions for computing
    #__________________________________________________________________

    # Visualization Methods
    #__________________________________________________________________
    def plot_param(self):
        """  
        Plot all functions in the parameter list along the x axis of
        a 3d plot.
        """
        # setup domain and plot shift
        labels = ['emission', 'absorption', 'scattering']
        values = [self['emission'], self['absorb'], self['scat']]
  
        plot_grid_list(self.x, self.y, self.z, values, labels)


    def plot_fluxes(self, *args):
        """ plot all the fluxes stored in self """
        nargin = len(args)
        
        labels = ['flux density', 'flux vector']
        values = [self['flux'], 
                  sp.array([self['xflux'], self['yflux'], self['zflux']])]

        plot_grid_list(self.x, self.y, self.z, values, labels)
        
    def plot_sos(self):
        """ plot successive orders of scattering """
        labels = ['ballistic'+repr(self['ballistic'].max())[0:6], 
                  'single'+ repr(self['single'].max())[0:6], 
                  'double' + repr(self['double'].max())[0:6], 
                  'multiple'+repr(self['multiple'].max())[0:6]]
        values = [self['ballistic'], self['single'], self['double'], self['multiple']]
        
        plot_grid_list(self.x, self.y, self.z, values, labels)

    def plot_sig_and_source(self):
        labels = ['grid sigma', 'grid source']
        values = [self.sigma, self.source]

        plot_grid_list(self.x, self.y, self.z, values, labels)


#         # plot flux
#         if nargin == 2:
#             mlab.figure(args[0])
#             mlab.clf()
#         else:
#             mlab.figure()

#         # plot routine
#         x, y, z = self.pos[0], self.pos[1], self.pos[2]
#         m = self.supp
#         mlab.points3d(x[m], y[m], z[m], self['flux'][m])
#         mlab.colorbar(orientation='vertical')
#         mlab.axes()
#         mlab.orientation_axes()
#         mlab.title('flux')

#         # plot vector flux 
#         if nargin == 2:
#             mlab.figure(args[1])
#             mlab.clf()
#         else:
#             mlab.figure()

#         # plot routine
#         m = self.supp
#         mlab.quiver3d(x[m], y[m], z[m], 
#                       self['xflux'][m], self['yflux'][m], self['zflux'][m])
#         mlab.colorbar(orientation='vertical'), mlab.colorbar(orientation='vertical')
#         mlab.title('vector flux')
#         mlab.axes()
#         mlab.orientation_axes()


# Subclass Domain to define Rays
#__________________________________________________________________
class Rays(Domain):
    """ Subclass Domain to add ray definition, sample and resample, and
    ode-solver methods. """
    def __init__(self, name, opt):
        """ initialize all members """
        super(Rays, self).__init__(name, opt)
        self.M = opt.M
        # initialize rays and function values as zero
        self.rays = sp.zeros(opt.rays_shape)
        self.grid_rad = sp.zeros(opt.grid_val_shape)
        for key in opt.rays_key_list:
            self[key] = sp.zeros(opt.rays_val_shape)
        # set solver object solver
        self.solver = solver.get_ray_rad
        # placehold inoterp and solver
        self.ds = None
        self.interp = None      

        
    # methods for setting members
    #__________________________________________________________________
    # new ds setting methods 
    def set_new_ds(self, ds):
        """ Set a new direction by redefining rays and setting interp
        object members for interpolation. """
        self.ds = ds
        self.set_rays(ds)
        # instantiate and adopt interpolation functions
        #self.interp = interp.MapCoords('Rays.interp', self)
        if self.interp == None:
            self.interp = interp.WeightsNeighbors('Rays.interp', self)
        else:
            self.interp.set_new_rays(self)
        self.get_grid_vals = self.interp.get_grid_vals_from
        self.get_rays_vals = self.interp.get_rays_vals_from

    def set_rays(self, ds):
        """ set the rays attribute for this particular ds """
        x, y, z = ds.vdir[0], ds.vdir[1], ds.vdir[2];        
        s = sp.sqrt(1.0-z**2)
        rot = sp.array([[x, y, z], [-y/s, x/s, 0.0], [-x*z/s, -y*z/s, s]]).transpose()
        self.rot = rot
        # Rotate all points in space
        self.rays = sp.tensordot(rot, self.pos, axes=(-1,0))
        self.opp_rays = sp.tensordot(rot.transpose(), self.pos, axes=(-1,0))

    # interpolation 
    def sample_from(self, grid):
        """ use interpolation object to rample grid data from grid.  
        I will need sigma and emission data """
        self['sigma'] = sp.where(grid.supp, self.get_rays_vals(grid.sigma), 0.0)
        self['source'] = sp.where(grid.supp, self.get_rays_vals(grid.source), 0.0)

    def resample_grid_rad(self):
        """ resample new data and store as an member, Rays.resample_rad 
        """
        self.grid_rad = sp.where(self.supp, self.get_grid_vals(self['rad']), 0.0)

    def solve_ballistic(self, income):
        """ propigate the ballistic light """
        self['rad'] = self.solver(self, income)

    def solve_scattering(self):
        """ solve for intensity during a certain order of scattering """
        self['rad'] = self.solver(self, sp.zeros(self.rays_shape[2:]))

        
    # functions for computing
    #__________________________________________________________________
    

### ====================================================================
#   Unittests
### ====================================================================

if __name__ == '__main__':
    import unittest
    import sphere
    import group 

    # Set global variables for unittesting
    N, M = 19, 19
    group_name = 'group24'
    o = opt.Options('options', N, M, group_name)
    g = group.Group(o)
    a = sphere.AllDirs('ad', o, g)
    ds = a.ds_list[0]

    
    # instantiate grid object
    grid = Grid('grid', o)
    rays = Rays('rays', o)
    # attr lookup
    grid_data = ['sigma', 'source', 'loop_flux', 'loop_vflux', 'act', 'ds', 
                 'scat_order', 'pos', 'supp'] # needed data attributes 
    rays_data = ['N', 'M', 
                 'rays', 'pos', 'supp', 
                 'ds', 
                 'interp', 'solver',
                 'grid_rad']
    # grid_acted_on_by lookup
    ga_dict = {'':(0, 1, 2), 
               'a':(-1, 1, 2), 
               'b':(0, -2, 2), 
               'c':(0, 1, -3),
               'd':(0, 2, 1),
               'e':(2, 1, 0),
               'f':(1, 0, 2),
               's':(2, 0, 1)}
    # update flux lookup
    flux_list = ['flux', 'xflux', 'yflux', 'zflux',
                 'ballistic', 'single', 'double', 'multiple']
    # Plot stuff for verification
    mlab.figure(1)
    mlab.clf()
    grid.plot_param()

    # instantiate grid object
    rays = Rays('rays', o)
    rays.set_new_ds(ds)

    
#     # plot sig and sources for several actions to see if they are being rotated correctly

#     grid.setup_scatter_loop()
#     grid.set_new_ds(ds)

#     for act in g:
#         grid.plot_sig_and_source()
#         grid.set_new_act(act)
#         rand_rad = sp.random.random(grid.loop_flux.shape)
#         grid.update_loop_fluxes(rand_rad)
#         grid.plot_sig_and_source()
    
        


    #define a convenience error function for any two ndarrays
    def get_error(f, g):
        """ 
        define reletive L1 error with g true 
        """
        return (abs(f - g)).sum() / abs(g).sum()

    # Unittest for 
    class TestDomain(unittest.TestCase):
        """
        Test object for the Grid and Ray sub-classes of Domain. 
        """
        def setUp(self):
            """ 
            Prepare for each test. 
            """
            # defint grid and ray objects
            self.g = Grid('grid', o) # inst. grid obj
            self.r = Rays('rays', o) # inst. ray obj
            self.r.set_new_ds(ds)
            self.g.set_new_ds(ds)
            # set grid and ray positions for defining vals
            x, y, z = (self.r.pos[0], 
                       self.r.pos[1], 
                       self.r.pos[2])
            xrays, yrays, zrays = (self.r.rays[0], 
                                   self.r.rays[1], 
                                   self.r.rays[2])
            # define grid_vals and rays_vals
            self.grid_vals = param.get_values(x, 
                                              y, 
                                              z, 'ybump')
            self.rays_vals = param.get_values(xrays, 
                                              yrays, 
                                              zrays, 'ybump')


        # Test data existance and shape
        #_________________________________________________________________
        # attribute existance
        def test_rays_attribute(self):
            """ grid and rays attribute existance """
            # rays attr
            for attr in rays_data:
                self.assert_(hasattr(self.r, attr),
                             "rays missing attribute,  '"+attr+"'")
            # grid attr
            for attr in grid_data:
                self.assert_(hasattr(self.g, attr), 
                             "grid missing attribute , '"+attr+"'")
        # ndarray attribute shape check
        def test_shape(self):
            """ Domain members have shapes given by options. """
            # grid position arrays correct shape (for Rays.pos too)
            self.assertEqual(self.g.pos.shape, 
                             tuple(o.pos_shape))
            self.assertEqual(self.r.pos.shape,
                             tuple(o.pos_shape))         
            # test that grid stores value arrays with correct shape 
            for key in o.grid_key_list:
                self.assert_(self.g.has_key(key))
                self.assertEqual(self.g[key].shape, 
                                 tuple(o.grid_val_shape))
            # rays.vrays the correct shape:
            self.assertEqual(self.r.rays.shape, 
                             tuple(o.rays_shape))
            for key in o.rays_key_list:
                self.assert_(self.r.has_key(key))
                self.assertEqual(self.r[key].shape, 
                                 tuple(o.rays_val_shape))
        # Check the key existance and value shapes for grid/rays
        def test_values(self):
            """ Keys and values of grid and ray """
            # value arrays are of correct shape
            for key in o.grid_key_list:
                self.assert_(key in self.g.keys())
                self.assertEqual(self.g[key].shape, 
                                 tuple(o.grid_val_shape))
            # value arrays are of correct shape
            for key in o.rays_key_list:
                self.assertEqual(self.r[key].shape, 
                                 tuple(o.rays_val_shape))


        # Test Domain methods that set members
        #_________________________________________________________________

        def test_ray_methods(self):
            """ Method for resetting ray members"""
            # test for method setting a new ray direction
            self.r.set_new_ds(ds)
            self.assert_(self.r.ds==ds)
            # test for methods sampling from grid
            self.g.sigma = self.grid_vals
            self.g.source = self.grid_vals
            self.r['rad'] = self.rays_vals
            self.r.sample_from(self.g)
            self.r.resample_grid_rad()
            # define true sigma and calculate error 
            e1 = get_error(self.r['sigma'], self.rays_vals)
            self.assertAlmostEqual(e1, 0.0, 1)
            e2 = get_error(self.r['source'], self.rays_vals)
            self.assertAlmostEqual(e2, 0.0, 1)
            e3 = get_error(self.r.grid_rad, self.grid_vals)
            self.assertAlmostEqual(e3, 0.0, 1)
            print 'max interp error = ', sp.array([e1, e2, e3]).max()
            # test solving methods 
            self.r.solve_ballistic(sp.zeros((N, N)))
            self.r.solve_scattering()
        
        def test_grid_methods(self):
            """ test execution of loop """
            # test group action method agains lookup
            grid.setup_scatter_loop()
            grid.set_new_ds(ds)
            grid.source = self.grid_vals
            grid.sigma = self.grid_vals
            rand_rad = sp.random.random(grid.loop_flux.shape)
            grid.plot_sig_and_source()

            for act in g: 
                grid.set_new_act(act)
                grid.update_loop_fluxes(rand_rad)
                assert (grid.source == self.grid_vals).all(), "source problem with, " + act
                assert (grid.sigma == self.grid_vals).all(), "sigma problem with, " + act
            grid.plot_sig_and_source()
                
#             for act in g:
#                 grid.set_new_act(act)
#                 rand_rad = sp.random.random(grid.loop_flux.shape)
#                 grid.update_loop_fluxes(rand_rad)
#             grid.store_loop_fluxes()






        # Test signitures of Domain functions
        #_________________________________________________________________
    
    # run unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDomain)
    unittest.TextTestRunner(verbosity=2).run(suite)









