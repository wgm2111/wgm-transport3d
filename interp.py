### ====================================================================
###  Python-file
###     author:          William G. K. Martin
###     filename:        interp.py
### ====================================================================


import scipy as sp
import scipy.ndimage as ndim
map_coordinates = ndim.map_coordinates

# Define functions 
#__________________________________________________________________
lagrange = []
lagrange.append(lambda x, y, z: (1 - z) * (1 - x - y))
lagrange.append(lambda x, y, z: (1 - z) * (x))
lagrange.append(lambda x, y, z: (1 - z) * (y))
lagrange.append(lambda x, y, z: (z) * (1 - x - y))
lagrange.append(lambda x, y, z: (z) * (x))
lagrange.append(lambda x, y, z: (z) * (y))


def get_coords(x, y, z, tx, ty, tz, supp):
    """ 
    Map (x, y, z) to fractional indecies on the target grid (tx, ty, tz).
    """
    x0, y0, z0 = x[0, 0, 0], y[0, 0, 0], z[0, 0, 0]
    dx = x[1, 0, 0] - x0
    dy = y[0, 1, 0] - y0
    dz = z[0, 0, 1] - z0
    ivals, jvals, kvals = sp.zeros(x.shape), sp.zeros(x.shape), sp.zeros(x.shape)
    ivals[supp] = (tx[supp] - x0) / dx
    jvals[supp] = (ty[supp] - y0) / dy
    kvals[supp] = (tz[supp] - z0) / dz
    return ivals, jvals, kvals

    
# Interpolation Class
#__________________________________________________________________
class Interp(object):
    """ Class for storing members needed for computing neigh_strides """
    def __init__(self, name, rays_obj):
        r = rays_obj            # redefine for readability
        # spatial coordinates from rays
        self.pos = r.pos
        self.h = r.grid_stride
        self.rays = r.rays
        self.opp = r.opp_rays
        # other nessesary members
        self.rot = r.rot
        self.name = name
        self.supp =r.supp




class MapCoords(Interp):   
    """
    use ndimage.map_coordinates interpolation to transfer to and from rays
    """
    def __init__(self, name , rays_obj):
        """ 
        Setup interpolant by getting coordinates 
        """
        super(MapCoords, self).__init__(name, rays_obj)
        x, y, z = self.pos[0], self.pos[1], self.pos[2]
        
        # compute coords needed to map values to grid
        self.coords_2rays = get_coords(x, y, z, 
                                       self.rays[0], 
                                       self.rays[1], 
                                       self.rays[2], self.supp)
        self.coords_2grid = get_coords(x, y, z, 
                                       self.opp[0], 
                                       self.opp[1], 
                                       self.opp[2], self.supp)

    def get_rays_vals_from(self, vals):
        """ use coords to find the ray vals """
        return map_coordinates(vals, self.coords_2rays, order=1)
        
    def get_grid_vals_from(self, vals):
        """ use coords to find the grid vals """
        return map_coordinates(vals, self.coords_2grid, order=1)


class WeightsNeighbors(Interp):
    """
    Use my own trilinear interpolant to compute values 
    """
    def __init__(self, name , rays_obj):
        """ 
        Setup interpolant by computing weights and neighbors 
        """
        super(WeightsNeighbors, self).__init__(name, rays_obj)
        # define standard stride to find neighbors withough correcting for element orientation
        self.nsteps = sp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], 
                                [0, 0, 1], [1, 0, 1], [0, 1, 1]])
        self.nsteps_flipped = sp.array([[1, 1, 0], [0, 1, 0], [1, 0, 0], 
                                        [1, 1, 1], [0, 1, 1], [1, 0, 1]])        

        # Stuff for sampling
        self.ref, self.rem = self.get_refrem(self.rays[0],
                                             self.rays[1],
                                             self.rays[2],
                                             self.supp)
        # local coordinate values and steps to find neighbors from the reference neighbor
        self.loc, self.xneigh2rays, self.yneigh2rays, self.zneigh2rays = self.get_loc_and_neigh()
        # define weights
#        self.lagrange = self.get_lagrange()        
        self.weights2rays = self.get_weights()

        # repeat for neigh2grid and weights2grid
        self.ref, self.rem = self.get_refrem(self.opp[0],
                                             self.opp[1], 
                                             self.opp[2], 
                                             self.supp)
        # local coordinate values and steps to find neighbors from the reference neighbor
        self.loc, self.xneigh2grid, self.yneigh2grid, self.zneigh2grid = self.get_loc_and_neigh()
        # define weights
        self.weights2grid = self.get_weights()
        
    def set_new_rays(self, rays_obj):
        """ set new rays without instantiating """
        r = rays_obj            # redefine for readability
        # spatial coordinates from rays
        self.pos = r.pos
        self.h = r.grid_stride
        self.rays = r.rays
        self.opp = r.opp_rays
        # other nessesary members
        self.rot = r.rot
        self.supp =r.supp
        # Stuff for sampling
        self.ref, self.rem = self.get_refrem(self.rays[0],
                                             self.rays[1],
                                             self.rays[2],
                                             self.supp)
        # local coordinate values and steps to find neighbors from the reference neighbor
        self.loc, self.xneigh2rays, self.yneigh2rays, self.zneigh2rays = self.get_loc_and_neigh()
        # define weights
#        self.lagrange = self.get_lagrange()        
        self.weights2rays = self.get_weights()

        # repeat for neigh2grid and weights2grid
        self.ref, self.rem = self.get_refrem(self.opp[0],
                                             self.opp[1], 
                                             self.opp[2], 
                                             self.supp)
        # local coordinate values and steps to find neighbors from the reference neighbor
        self.loc, self.xneigh2grid, self.yneigh2grid, self.zneigh2grid = self.get_loc_and_neigh()
        # define weights
        self.weights2grid = self.get_weights()


    def get_rays_vals_from(self, vals):
        """ use coords to find the ray vals """
        # initilize rays_vals
        rv = sp.zeros(vals.shape)
        # loop over all neighbors and weights 
        for k in range(6):#self.interp.weights.shape[0]):
            weights = self.weights2rays[k,...]
            xneigh = self.xneigh2rays[k,...]
            yneigh = self.yneigh2rays[k,...]
            zneigh = self.zneigh2rays[k,...]
            rv += weights * vals[xneigh, yneigh, zneigh] 
        return rv
        
        
    def get_grid_vals_from(self, vals):
        """ use coords to find the grid vals """
        # initilize rays_vals
        gv = sp.zeros(vals.shape)
        # loop over all neighbors and weights 
        for k in range(6):#self.interp.weights.shape[0]):
            weights = self.weights2grid[k,...]
            xneigh = self.xneigh2grid[k,...]
            yneigh = self.yneigh2grid[k,...]
            zneigh = self.zneigh2grid[k,...]
            gv += weights * vals[xneigh, yneigh, zneigh] 
        return gv

    def get_refrem(self, x, y, z, supp):
        """ Get the reference neighbor indicies and and polinomial remainders. """
#         dx = x[1,0,0] - x[0,0,0]
#         dy = y[0,1,0] - y[0,0,0]
#         dz = z[0,0,1] - z[0,0,0]

        dial0 = sp.where(supp,(x + 1) / self.h[0], 0)  #2 * (self.shape[0] - 1)
        dial1 = sp.where(supp,(y + 1) / self.h[1], 0)  #2 * (self.shape[1] - 1)
        dial2 = sp.where(supp,(z + 1) / self.h[2], 0)  #2 * (self.shape[2] - 1) 
        #Calculate the index of allong the rays 
        ref_0 = sp.where(supp, sp.cast[int](dial0), 0)
        ref_1 = sp.where(supp, sp.cast[int](dial1), 0)
        ref_2 = sp.where(supp, sp.cast[int](dial2), 0)
        # Calculate the remainder and logical arrays
        rem_0 = sp.where(supp, dial0 - ref_0, 0)
        rem_1 = sp.where(supp, dial1 - ref_1, 0)
        rem_2 = sp.where(supp, dial2 - ref_2, 0)
        return sp.array([ref_0, ref_1, ref_2]), [rem_0, rem_1, rem_2]


    def get_loc_and_neigh(self):
        """ Return the appropriate local coordinates and neighbor steps """
        # define a logical array for indicating that a element flip in nessecay
        log_flip = self.rem[0] + self.rem[1] >= 1
        self.log_flip = log_flip
        # define local coordinates
        xloc = sp.where(log_flip, 1.0 - self.rem[0], self.rem[0])
        yloc = sp.where(log_flip, 1.0 - self.rem[1], self.rem[1])
        zloc = self.rem[2]
    
        # define steps to take to find a neighbor from the reference neighbor
        nstep_shape = (6,) + log_flip.shape
        xnstep, ynstep, znstep = (sp.zeros(nstep_shape, dtype=int), 
                                  sp.zeros(nstep_shape, dtype=int), 
                                  sp.zeros(nstep_shape, dtype=int))
        for k in range(6):
            xnstep[k, ...] = sp.where(log_flip, self.nsteps_flipped[k, 0], self.nsteps[k, 0])
            ynstep[k, ...] = sp.where(log_flip, self.nsteps_flipped[k, 1], self.nsteps[k, 1])
            znstep[k, ...] = sp.where(log_flip, self.nsteps_flipped[k, 2], self.nsteps[k, 2])
        
        # calculate the neighbors by adding the reference index 
        xneigh = sp.where(self.supp, xnstep + self.ref[0][...], 0)
        yneigh = sp.where(self.supp, ynstep + self.ref[1][...], 0)
        zneigh = sp.where(self.supp, znstep + self.ref[2][...], 0)

        return (xloc, yloc, zloc), xneigh, yneigh, zneigh

    def get_weights(self):
        """ define the weights using the local coordinates """
        weights = sp.zeros((6,) + self.supp.shape)
        for k in range(6):
            weights[k, ...] = lagrange[k](self.loc[0], self.loc[1], self.loc[2])
        return weights


### ====================================================================
#   Unittests
### ====================================================================

if __name__ == '__main__':


    # imports for tesing
    import sys
    import unittest
    import options, group, sphere, domain
    import parameter as param


    # Set global variables for unittesting
    N, M = 60, 30
    group_name = 'group48'
    o = options.Options('options', N, M, group_name)
    g = group.Group(o)
    vdir = sp.array([1, .3, .7])
    ds = sphere.DS('ds', o, g, 1, vdir)
    
    # instantiate grid object
    rays = domain.Rays('rays', o)
    rays.set_new_ds(ds)
    # set grid and ray positions for defining vals
    x, y, z = rays.pos[0], rays.pos[1], rays.pos[2]
    xrays, yrays, zrays = rays.rays[0], rays.rays[1], rays.rays[2]
    # define grid_vals and rays_vals
    grid_vals = param.get_values(x, y, z, 'icosahedron')
    rays_vals = param.get_values(xrays, yrays, zrays, 'icosahedron')
    my_interp = WeightsNeighbors('my_interp', rays)
            
    def get_error(f, g):
        """ define reletive L1 error with g true """
        return (abs(f - g)).sum() / abs(g).sum()
    
    # define the unitest object to compare the true results above with 
    class TestDomain(unittest.TestCase):
        """ compare interp values with true ones """
        def setUp(self):
            """ instantiate and interpolation object """
            self.interp = MapCoords('interp', rays)
            self.my_interp = WeightsNeighbors('my_interp', rays)

        def test_get_grid_vals_from(self):
            """ calculate grid values and compare error """
            grid_interp = self.interp.get_grid_vals_from(rays_vals)
            error = get_error(grid_interp, grid_vals)
            print 'rel_L1error = ', error

        def test_get_rays_vals_from(self):
            """ calculate rays values and compare error """
            rays_interp = self.interp.get_rays_vals_from(grid_vals)
            error = get_error(rays_interp, rays_vals)
            print 'rel_L1error = ', error
        
        def test_my_get_grid_vals_from(self):
            """ compare error for my_interp grid2rays """
            grid_interp = self.my_interp.get_grid_vals_from(rays_vals)
            error = get_error(grid_interp, grid_vals)
            print 'rel_L1error = ', error

        def test_my_get_rays_vals_from(self):
            """ compare error for my_interp rays2grid """
            rays_interp = self.my_interp.get_rays_vals_from(grid_vals)
            error = get_error(rays_interp, rays_vals)
            print 'rel_L1error = ', error



    # run unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDomain)
    unittest.TextTestRunner(verbosity=2).run(suite)










