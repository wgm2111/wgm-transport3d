### ====================================================================
###  Python-file
###     author:          William G. K. Martin
###     filename:        options.py
###     version:         
###     created:         
###       on:            
###     last modified:   
###       at:            
###     URL:             
###     email:           
###  
### ====================================================================



### ====================================================================
#   Import stuff
### ====================================================================
import scipy as sp
import enthought.mayavi.mlab as mlab

### ====================================================================
#   Define mixin classes
### ====================================================================
class Options(object):
    """ This class will be the parent of mixins for my transport simulation
    and should contain attributes that all mixins will need.
    """
    pass



### DomainMixIn class ### ==============================================================

#---------------------------------------------------------------------------------------
# DomainMixIn class - backpack of supplies for all Domain objects (Grid,Ray,Par,Sol).
#---------------------------------------------------------------------------------------
class DomainMixIn(Options):
    """
    Set all the parameters that will be required of the simulation.
    grid_shape
    ray_shape
    grid_strides
    ray_strides
    """
    def __init__(self, name, N):
        """ set the parameters """
        self.name = name
        self.grid_shape = sp.array([N, N, N])
        self.shape = self.grid_shape
        self.ray_shape = sp.array([N, N, N])
        self.grid_stride = (2.0 / (self.grid_shape[0]-1.0), 2.0 / (self.grid_shape[1]-1.0), 2.0 / (self.grid_shape[2]-1.0))
        self.ray_stride = (2.0 / (self.ray_shape[0]-1.0), 2.0 / (self.ray_shape[1]-1.0), 2.0 / (self.ray_shape[2]-1.0))
        h = self.grid_stride
        self.x, self.y, self.z = sp.ogrid[-1.0:1.0+h[0]/2:h[0], -1.0:1.0+h[1]/2:h[1], -1.0:1.0+h[2]/2:h[2]]
        self.x, self.y, self.z = sp.broadcast_arrays(self.x, self.y, self.z)
        self.grid_radius = aux_r = sp.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.supp_rad = 1 - 2.5 * h[0] / 4.0
        self.grid_supp = self.grid_radius < self.supp_rad

    # make a dictionary of standard functions with their **fun_option dictionaries
    def evaluate(self, x, y, z, fun_type='xbump'):
        return get_values(x, y, z, self.shape, self.grid_supp, fun_type)
        
    # routine for plotting a frame around 
    def plot_box(self):
        v = sp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        v[v==0]=-1
        i = sp.array([0, 1, 5, 3, 0, 2, 4, 1, 5, 7, 4, 2, 6, 7, 6, 3, 0]); 
        mlab.plot3d(v[i,0], v[i,1], v[i,2], tube_radius=.01)
        mlab.axes()
    # Routine for plotting values at gridpoints 
    def plot_grid_values(self, x, y, z, values):
        """ Plot the values given along rays """
        val_min = values[self.supp].min()
        val_max = values[self.supp].max()
        supp = abs(values) >= val_max / 100
        mlab.points3d(x[supp], y[supp], z[supp], values[supp], vmin=val_min, vmax=val_max)
        

#---------------------------------------------------------------------------------------
# GridMixIn class - a backpack with supplies for Grid and children
#---------------------------------------------------------------------------------------
class GridMixIn(Options):
    """
    Set all member variables that need to be flexible 
    """
    def __init__(self, name):
        """ Basic init file for grid specifics """
        self.name = name

#---------------------------------------------------------------------------------------
# RayMixIn class - a backpack with supplies for Rays and children
#---------------------------------------------------------------------------------------
class RaysMixIn(Options):
    """
    Set important ray only members and methods.
    """
    def __init__(self, name, dom, direction = None):
        """ Define ray specific parameters """
        # set the default shape to that of the rays
        self.name = name
        self.shape = dom.ray_shape
        # Define the rotation matrix
        try: 
            self.direction = normalize(direction)
            x, y, z = self.direction[0], self.direction[1], self.direction[2];        s = sp.sqrt(1.0-z**2)
            R_dir = sp.array([[x, y, z], [-y/s, x/s, 0.0], [-x*z/s, -y*z/s, s]]).transpose()
        except: raise Exception('RaysMixIn.__init__(self, name, domain_obj_instance, direction)')
        # Rotate all points in space
        self.rotation_matrix = R_dir
        self.rays = sp.tensordot(R_dir, sp.broadcast_arrays(dom.x, dom.y, dom.z), axes=(-1,0))
        self.xrays, self.yrays, self.zrays = self.rays[0,...], self.rays[1,...], self.rays[2,...]
        self.ray_rad = norm(self.xrays, self.yrays, self.zrays)
        self.ray_supp = self.ray_rad < dom.supp_rad
        self.supp = self.ray_supp

        # Set weights and neighbors in the standard way
        # self.weights, self.neigh = self.get_weights_and_neigh(self.xrays, self.yrays, self.zrays, dom.grid_stride)

        # Define an interp object
        self.interp = Interp('interp', self.xrays, self.yrays, self.zrays, self.supp, dom.grid_stride, interp_method='prism')

    def update_interp(self):
        """ Method responsible for updating the interp object to account for changes in rays """
        self.interp.update('interp', self.xrays, self.yrays, self.zrays, self.supp, dom.grid_stride, interp_method='prism')

    def plot_ray_values(self, values):
        """ Plot the values given along rays """
        val_min = values[self.ray_supp].min()
        val_max = values[self.ray_supp].max()
        step = sp.array([self.shape[1] / 6 + 1, self.shape[2] / 6 + 1])
        for i in range(self.shape[1]):
            if i % step[0] == 0:
                for j in range(self.shape[2]):
                    if j % step[1] == 0:
                        p_log = self.ray_rad[:, i, j] < 1
                        if any(p_log) is True:
                            xx = self.xrays[p_log, i, j]
                            yy = self.yrays[p_log, i, j]
                            zz = self.zrays[p_log, i, j]
                            vv = values[p_log, i, j]
                            mlab.plot3d(xx, yy, zz, vv,
                                        tube_sides=6, tube_radius=0.01, vmin=val_min, vmax=val_max)


### ====================================================================
#   Define a short library of functions to be mixed
### ====================================================================
    
#---------------------------------------------------------------------------------------
# Interpolation Class
#---------------------------------------------------------------------------------------

class Interp(object):
    """ Class for storing members needed for computing neigh_strides """
    def __init__(self, name, x, y, z, supp, gstride, interp_method='prism'):
        self.name = name +', ' + interp_method
        self.supp = supp
        self.ref, self.rem = self.get_refrem(x, y, z, supp, gstride)

        # define standard stride to find neighbors withough correcting for element orientation
        self.nsteps = sp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], 
                                [0, 0, 1], [1, 0, 1], [0, 1, 1]])
        self.nsteps_flipped = sp.array([[1, 1, 0], [0, 1, 0], [1, 0, 0], 
                                        [1, 1, 1], [0, 1, 1], [1, 0, 1]])
        
        # local coordinate values and steps to find neighbors from the reference neighbor
        self.loc, self.xneigh, self.yneigh, self.zneigh = self.get_loc_and_neigh()

        # define lagrange anonymous functions
        self.lagrange = self.get_lagrange()        
        # define weights
        self.weights = self.get_weights()


    def update(self, name, x, y, z, supp, gstride):
        """ update the instance with new information """
        self.__init__(name, x, y, z, supp, gstride)

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
        xnstep, ynstep, znstep = sp.zeros(nstep_shape, dtype=int), sp.zeros(nstep_shape, dtype=int), sp.zeros(nstep_shape, dtype=int)
        for k in range(6):
            xnstep[k, ...] = sp.where(log_flip, self.nsteps_flipped[k, 0], self.nsteps[k, 0])
            ynstep[k, ...] = sp.where(log_flip, self.nsteps_flipped[k, 1], self.nsteps[k, 1])
            znstep[k, ...] = sp.where(log_flip, self.nsteps_flipped[k, 2], self.nsteps[k, 2])
        
        # calculate the neighbors by adding the reference index 
        xneigh = xnstep + self.ref[0][...]
        yneigh = ynstep + self.ref[1][...]
        zneigh = znstep + self.ref[2][...]

        return (xloc, yloc, zloc), xneigh, yneigh, zneigh

        
    def get_weights(self):
        """ define the weights using the local coordinates """
        weights = sp.zeros((6,) + self.supp.shape)
        for k in range(6):
            weights[k, ...] = self.lagrange[k](self.loc[0], self.loc[1], self.loc[2])
        return weights

    def get_refrem(self, x, y, z, supp, gstride):
        """ Get the reference neighbor indicies and and polinomial remainders. """
        dial0 = sp.where(supp,(x + 1) / gstride[0], 0)  #2 * (self.shape[0] - 1)
        dial1 = sp.where(supp,(y + 1) / gstride[1], 0)  #2 * (self.shape[1] - 1)
        dial2 = sp.where(supp,(z + 1) / gstride[2], 0)  #2 * (self.shape[2] - 1) 
        #Calculate the index of allong the rays 
        ref_0 = sp.where(supp, sp.cast[int](dial0), 0)
        ref_1 = sp.where(supp, sp.cast[int](dial1), 0)
        ref_2 = sp.where(supp, sp.cast[int](dial2), 0)
        # Calculate the remainder and logical arrays
        rem_0 = sp.where(supp, dial0 - ref_0, 0)
        rem_1 = sp.where(supp, dial1 - ref_1, 0)
        rem_2 = sp.where(supp, dial2 - ref_2, 0)
        return sp.array([ref_0, ref_1, ref_2]), [rem_0, rem_1, rem_2]

    def get_lagrange(self):
        """ get Lagrange list of functions """
        lagrange = []
        lagrange.append(lambda x, y, z: (1 - z) * (1 - x - y))
        lagrange.append(lambda x, y, z: (1 - z) * (x))
        lagrange.append(lambda x, y, z: (1 - z) * (y))
        lagrange.append(lambda x, y, z: (z) * (1 - x - y))
        lagrange.append(lambda x, y, z: (z) * (x))
        lagrange.append(lambda x, y, z: (z) * (y))
        return lagrange


    
#---------------------------------------------------------------------------------------
# Functions for defining values
#---------------------------------------------------------------------------------------

def get_values(x, y, z, shape, supp, fun_type):
    """ Use the dictionary **fun_options to define values for smooth_bump functions 
    and step functions supported on the interior of small spheres """
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
    #---------------------------------------------------------------------------------------

    # Set the options for this function call based on the input fun_type
    try:
        smooth = fun_dict[fun_type]['smooth']
        centers = fun_dict[fun_type]['centers']
        radii = fun_dict[fun_type]['radii']
        normalize = fun_dict[fun_type]['normalize']
        scale_factor = fun_dict[fun_type]['scale_factor']
    except: print "fun_type not vaild, use: ", fun_dict.keys()

    # Initialize function values
    values = sp.zeros(shape)

    # Loop over all bumps and steps to build values
    for i in range(len(smooth)):
        if smooth[i] is True:
            bump = sp.where(supp, smooth_bump(x, y, z, centers[i], radii[i]), 0.0)
            values = values + scale_factor[i] * bump
        else:
            values = sp.where(supp, values + scale_factor[i] * step(x, y, z, centers[i], radii[i]), 0.0)

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
#   Test this code using a conditional main statement
### ====================================================================

if __name__ == '__main__':
    """ script for testing mixins """
    
    plotting = True

    test_Domain = True
    test_Grid = True
    test_Rays = True
    test_Interp = True
    
    # define a standard Options object
    op = Options()

    if test_Domain is True:
        d = DomainMixIn('DomainMixIn', 15)
        print "==============================================================="
        print d.name + ': test output '
        print "---------------------------------------------------------------"
        print 'grid_shape is ', d.grid_shape
        print 'ray_shape is ', d.ray_shape
        print 'grid_stride is ', d.grid_stride
        print 'ray_stride is ', d.ray_stride
        print "---------------------------------------------------------------"
        print 'DomainMixIn members:'
        print '\t', '%-15s \t %-15s' %('Member', 'Type')
        keylist = d.__dict__.keys()
        keylist.sort()
        for att in keylist:
            if att not in dir(op):
                print '\t', '%-15s' %(att), type(d.__dict__[att])
        print "---------------------------------------------------------------"
        if plotting is True:
            print "Plotting the radius as a function of position."
            xx, yy, zz = sp.broadcast_arrays(d.x, d.y, d.z)
            mlab.figure(1)
            mlab.clf()
#            mlab.points3d(xx, yy, zz, d.grid_radius)
            d.plot_box()
            mlab.colorbar(orientation='vertical')
            mlab.title('grid_radius')
            print "Plotting the radius again, but only on the unit ball"
            supp_plot = mlab.figure(2)
            mlab.clf()
#            mlab.points3d(xx[d.grid_supp], yy[d.grid_supp], zz[d.grid_supp], d.grid_radius[d.grid_supp])
            d.plot_box()
            mlab.colorbar(orientation='vertical')
            mlab.title('grid_radius on grid_supp')
            print "plotting the results of evaluate for fun type asymetric"
            evaluate_plot = mlab.figure(3)
            mlab.clf()
            values = d.evaluate(d.x, d.y, d.z, fun_type='icosahedron')
            values.max()
            supp = values >= .000001
            mlab.points3d(xx[supp], yy[supp], zz[supp], values[supp])              
            d.plot_box()
            mlab.colorbar(orientation='vertical')
            mlab.title('d.evaluate(..., fun_type=asymetric)')
        print "===============================================================", '\n'*2

    # Rays Testing
    if test_Rays is True:
        dom = DomainMixIn('DomainMixIn',7)
        d = RaysMixIn('RaysMixIn', dom, direction=sp.array([1.0, .4, .3]))
        print "==============================================================="
        print d.name + ': test output '
        print "---------------------------------------------------------------"
        print d.name + ' members:'
        print '\t', '%-15s \t %-15s' %('Member', 'Type')
        keylist = d.__dict__.keys()
        keylist.sort()
        for att in keylist:
            if att not in dir(op):
                print '\t', '%-15s' %(att), type(d.__dict__[att])
        print "---------------------------------------------------------------"
        print d.interp.name + ' members:'
        print '\t', '%-15s \t %-15s' %('Member', 'Type')
        keylist = d.interp.__dict__.keys()
        keylist.sort()
        for att in keylist:
            if att not in dir(object):
                print '\t', '%-15s' %(att), type(d.interp.__dict__[att])
        print "---------------------------------------------------------------"
        print '\n'
        sum = d.interp.weights.sum(axis=0)
        print " The wieght sum to a max of, ", sum.max(), " and a min of, ", sum.min()
        if plotting:
            print "*** Ploting RaysMixIn.ref and RaysMixIn.rem"
            mlab.figure(4)
            mlab.clf()
            xx, yy, zz = sp.broadcast_arrays(dom.x, dom.y, dom.z)
            mlab.points3d(xx[d.interp.ref], yy[d.interp.ref], zz[d.interp.ref], scale_factor = 0.05)
            mlab.quiver3d(xx[d.interp.ref], yy[d.interp.ref], zz[d.interp.ref], 
                          d.interp.rem[0], d.interp.rem[1], d.interp.rem[2], 
                          scale_factor=dom.grid_stride[0])
            mlab.points3d(d.xrays[d.supp], d.yrays[d.supp], d.zrays[d.supp], scale_factor=0.02, color=(1.0, 0.0, 0.0))
            dom.plot_box()
            

        print "---------------------------------------------------------------"
        # Interp Testing 
        if test_Interp is True:
            print " Plotting the neighbors of several points "
            mlab.figure(5)
            mlab.clf()
            xx, yy, zz = sp.broadcast_arrays(dom.x, dom.y, dom.z)
            
            # ray indicies
            raysl = (slice(3, 7, 2), slice(3, 7, 3), slice(3, 7, 3))
            
            # ray quantities for all neighbors
            sl = (slice(0,6),) + raysl; 
            refs = (0,) + raysl
            print refs
            
            # mappings from ray indicies to grid indicies 
            nx = d.interp.xneigh
            ny = d.interp.yneigh
            nz = d.interp.zneigh
            print "neighbor slice"
            
            # local coordinate system
            loc = d.interp.loc
            print yy[nx[refs], ny[refs], nz[refs]].shape 
            print loc[2][raysl].shape

            # logical array to indicate that an element has been flipped 
            log = d.interp.log_flip
            log.shape

            # assigning weights to each neighbor for each ray point
            weights = d.interp.weights
            
            # plot arrows pointing from the reference neighbor to the ray point 
            mlab.quiver3d(xx[nx[refs], ny[refs], nz[refs]], 
                          yy[nx[refs], ny[refs], nz[refs]], 
                          zz[nx[refs], ny[refs], nz[refs]], 
                          sp.where(log[raysl], -loc[0][raysl], loc[0][raysl]), 
                          sp.where(log[raysl], -loc[1][raysl], loc[1][raysl]), 
                          loc[2][raysl],  scale_factor=dom.grid_stride[0])
            # plot spheres to indicate each neighbors weight
            mlab.points3d(xx[nx[sl], ny[sl], nz[sl]], yy[nx[sl], ny[sl], nz[sl]], zz[nx[sl], ny[sl], nz[sl]], 
                          weights[sl])
            # plot black spheres to show the reference neighbors
            mlab.points3d(xx[d.interp.ref[0][raysl], d.interp.ref[1][raysl], d.interp.ref[2][raysl]], 
                          yy[d.interp.ref[0][raysl], d.interp.ref[1][raysl], d.interp.ref[2][raysl]], 
                          zz[d.interp.ref[0][raysl], d.interp.ref[1][raysl], d.interp.ref[2][raysl]], scale_factor = 0.05)

            
            # plot little spheres to show ray points
            mlab.points3d(d.xrays[raysl], d.yrays[raysl], d.zrays[raysl], color=(1, 0, 0), scale_factor=.05)
            dom.plot_box()
            mlab.title(" Neighbors and remainders ")            
            
