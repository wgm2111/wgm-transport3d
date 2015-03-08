# ====================================================================
#  Python-file
#     author:          William G. K. Martin
#     filename:        sphere.py
# ====================================================================
""" 
This sphere object was made to handle data storage and comutation tasks
that take place on the sphere.  Such responsibilities include calculating 
incoming boundary terms, and the weights used for integration.
"""

#   Import libraries and define classes
#___________________________________________________________________________

import sys

import scipy as sp
import scipy.integrate as spInt
import enthought.mayavi.mlab as mlab

import copy as copy
#import sphere_mixins as mi



# Library of mappinps defined on the sphere
#___________________________________________________________________________

def evaluate(dir, fun_type, center=None, waist=None):
    """ evaluate a particular scattering function at the specified directions """
    return get_values(dir, fun_type, y=center, c=waist)

# Incoming light from the boundary
def get_source_values(mu_p, c_p, mu_d, c_d, fun_type):
    """ mup is a ndarray of positional cosine values and cp is the corresponding henyey
    constant for the positional term.  mud and cd are both scalars and weight the 
    incoming intensity array by the direction """
    
    # initialize the sum
    if len(c_p) > 1:
        sum = sp.zeros(mu_p.shape[1:])  # when the first dimention runs over all beams
        for i, mu in enumerate(mu_p):
            sum += source_fun_dict[fun_type[i]](mu, c_p[i], mu_d[i], c_d[i])
                #henyey_lambda(mu, c_p[i]) * henyey_lambda(mu_d[i], c_d[i]) 
        return sum
    else:
        # when there is only one beam (no sum to take)
        return source_fun_dict[fun_type](mu_p, c_p, mu_d, c_d)

# scattering functions
henyey_lambda = lambda mu, c: 1.0 / 4.0 / sp.pi * (
    (1.0 - c**2.0) / (1.0 + c**2.0 - 2.0 * c * mu)**(3.0 / 2.0 ))
henyey_dict = {'fun_lambda':henyey_lambda,
               'const_dict':{'c':0.98, 'y':sp.array([1, 0, 0])}}

rayleigh_lambda = lambda mu: 3.0 / 16.0 / sp.pi * (1.0 + mu**2.0)
rayleigh_dict = {'fun_lambda':rayleigh_lambda,
                 'const_dict':{'y':sp.array([1, 0, 0])}}

isotropic_lambda = lambda mu: 1.0 / 4.0 / sp.pi
isotropic_dict = {'fun_lambda':isotropic_lambda,
                 'const_dict':{}}

fun_dict_lookup = {'isotropic':isotropic_dict, 'henyey':henyey_dict, 'rayleigh':rayleigh_dict}

# incoming boundary functions
henyey_source_lambda = lambda mup, cp, mud, cd: henyey_lambda(mup, cp) * henyey_lambda(mud, cd)
fanbeam_source_lambda = lambda mup, cp, mud, cd: henyey_lambda(mup, cp) * isotropic_lambda(mud)
plane_source_lambda = lambda mup, cp, mud, cd: isotropic_lambda(mup) * henyey_lambda(mud, cd)

source_fun_dict = {'doublehenyey':henyey_source_lambda,
                   'fanbeam':fanbeam_source_lambda,
                   'planeparallel':plane_source_lambda}

# Define a function for looking up the correct scattering function
def get_values(dir, fun_type, y=None, c=None):
    """ 
    Calculate the values of functions defined on the sphere.  ducktyping should 
    make this work for either single direction or multiple directions.  
    use the 'kwarg' y to specify a incoming direction for the calculation of the scattering angle.
    """
    # Look up parameters evaluating the inline functions above
    fun_dict = fun_dict_lookup[fun_type]
    fun_lambda = fun_dict['fun_lambda']
    const_dict = fun_dict['const_dict']

    # Calculate the cosine of the scattering angle
    if y: 
        mu = sp.tensordot(dir, y, axis=(0,0)) 
    else:
        if 'y' in const_dict.keys():
            mu = sp.tensordot(dir, const_dict['y'], axes=(0,0)) 
        else:
            mu = sp.tensordot(dir, sp.array([1, 0, 0]), axes=(0,0))

    # Return the values (specified function, specified directions)
    if c:
        return fun_lambda(mu, c)
    else:
        if 'c' in const_dict.keys():
            return fun_lambda(mu, const_dict['c'])
        else:
            return fun_lambda(mu)

# normalize a vector
def normalize(direction):
    length = sp.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
    return direction/length

# Calculate integration weights
def get_weights(M):
    """ Get the weights and positions of the directions on the sphere """
    h = 1/(sp.sqrt(3)*(M-.5))
    a = sp.linspace(h/4, 1/sp.sqrt(3)-h/4, M)

    # Define uniform gridpoints on the cube insribed in the unit sphere 
    # Open nodes in (x=1/sqr(3), y = 0..1/sqrt(3), z = 0..1/sqrt(3))
    A2, A3 = sp.meshgrid(a, a)
    A1 = 1/sp.sqrt(3)*sp.ones(A2.shape)

    # initialize an array for storing integrations weights for each direction
    weights = sp.zeros(A1.shape)
    density = weights.copy()
    density = weightDensity(A2, A3)*h**2.0
    j, i = sp.array([0, -1,1,1,-1, 0,1,0,-1]), sp.array([0, -1,-1,1,1, -1,0,1,0])
    lw = sp.array([4.0/9.0] + [1.0/36.0]*4 + [1.0/9.0]*4)

    for k in range(9):
        weights[1:M-1:1, 1:M-1:1] += lw[k] * density[1+i[k]:M-1+i[k]:1, 1+j[k]:M-1+j[k]:1] 

    for i in sp.arange(M):
        py, pz = a[i], a[0]
        (weights[i,0], err)  = spInt.dblquad(weightDensity, py-h/2, py+h/2, 
                                             lambda z: pz-h/2, lambda z: pz+h/2, epsabs = 1.0e-3)
        py, pz = a[i], a[-1]
        (weights[i,-1], err) = spInt.dblquad(weightDensity, py-h/2, py+h/2, 
                                             lambda z: pz-h/2, lambda z: pz+h/2, epsabs = 1.0e-3)
        py, pz = a[0], a[i]
        (weights[0, i], err) = spInt.dblquad(weightDensity, py-h/2, py+h/2, 
                                             lambda z: pz-h/2, lambda z: pz+h/2, epsabs = 1.0e-3)
        py, pz = a[-1], a[i]
        (weights[-1,i], err) = spInt.dblquad(weightDensity, py-h/2, py+h/2, 
                                             lambda z: pz-h/2, lambda z: pz+h/2, epsabs = 1.0e-3)

    #rescale the edges, each by 3/4 (corners by 9/16)
    weights[ 0, :] = 3.0 / 4.0 * weights[ 0, :]
    weights[-1, :] = 3.0 / 4.0 * weights[-1, :]
    weights[ :, 0] = 3.0 / 4.0 * weights[ :, 0]
    weights[ :,-1] = 3.0 / 4.0 * weights[ :,-1]

    # Define the directions on the unit sphere
    magnitude = sp.sqrt(A1**2 + A2**2 + A3**2)
    dir1 = A1 / magnitude
    dir2 = A2 / magnitude
    dir3 = A3 / magnitude
    
    # define weight mask
    weights_mask = weights > 0

    #return the directions and weights 
    return weights_mask, weights, sp.array([dir1, dir2, dir3])

def weightDensity(y, z):
    """
    Explicit functions defining the the weight density on the sphere
    """
    return sp.sqrt(1.0 / (1.0 / 3.0 + y**2 + z**2))**3.0 / (4.0 * sp.pi*sp.sqrt(3.0))


# Sphere class definition
#___________________________________________________________________________

class Sphere(dict):
    """ Sphere class will be responsible for storing and plotting functions
    on the unit sphere.  It will also prepare and provide access to a list
    of DS that will be used to iterate over.  Sphere is a metaclass, so it
    is not intended to be instantiated directly. """

    def __init__(self, name, options):
        """ instantiate with options """
        self.name = name
        self.shape = (options.M, options.M)          # store gridshape
        self.M = options.M                           # store gridlength
        # define evaluate to lookup functions in sphere.mixins
#         self.evaluate = mix.evaluate
        self.evaluate = evaluate

    # Define methods to return the components of the directions
    def xdir(self):
        return self.vdir[0]
    def ydir(self):
        return self.vdir[1]
    def zdir(self):
        return self.vdir[2]

    # Define action by group
    def acted_on_by(self, act):
        """ This function is uses ducktyping to work for any subclass
        of Sphere """
        new = copy.copy(self)
        # rotate the coorinates for spins
        if act.count('s') == 1:
            new.vdir = self.vdir[(2, 0, 1),...]
        elif act.count('s') == 2:
            new.vdir = self.vdir[(1, 2, 0),...]
        elif 'd' in act:
            new.vdir = self.vdir[(0, 2, 1),...]
        elif 'e' in act:
            new.vdir = self.vdir[(2, 1, 0),...]
        elif 'f' in act:
            new.vdir = self.vdir[(1, 0, 2),...]
        else:
            new.vdir = self.vdir[(0, 1, 2),...]
        # change signs of the directions components for flips
        if 'c' in act:
            new.vdir[2] = -new.vdir[2]
        if 'b' in act:
            new.vdir[1] = -new.vdir[1]
        if 'a' in act:
            new.vdir[0] = -new.vdir[0]
        return new
        

# AllDirs Class 
#___________________________________________________________________________

class AllDirs(Sphere):
    """ iterate over ds_list and plot scattering functions / weights on S2 """
    def __init__(self, name, options, g):
        super(AllDirs, self).__init__(name, options)
        self.name = name
        self.group_name = options.group_name
        self.weights_mask, self.weights, self.vdir = get_weights(self.M)        
#         self.weights_mask, self.weights, self.vdir = mix.get_weights(self.M)        
        self.maxWeight, self.minWeight = (self.weights).max(), (self.weights).min()

        # define all-direction arrays for storing/plotting values
        self.alldir, self.allweights = self.get_allarrays(g)
        self.alldvs = self.alldir * self.allweights
        # define a list of ds objects to iterate over
        self.ds_list = []
        for index, weight in enumerate(self.weights.flat):
            if self.weights_mask.flat[index]:
                ds = DS(str(index), options, g, weight,
                        sp.array([self.xdir().flat[index], 
                                  self.ydir().flat[index], 
                                  self.zdir().flat[index]]))
                self.ds_list.append(ds)
        
    def get_allarrays(self, g):
        """ Append arrays together to build arrays that contain all directions and or
        the coresponding function values """
        # Copy directions and weights
        xall, yall, zall  = self.vdir[0], self.vdir[1], self.vdir[2]
        allweights = self.weights[:]
        
        #loop to fill in for all directions
        for act in g:
            cp = self.acted_on_by(act)
            xall = sp.append(xall, cp.vdir[0], axis=0)
            yall = sp.append(yall, cp.vdir[1], axis=0)
            zall = sp.append(zall, cp.vdir[2], axis=0)
            allweights = sp.append(allweights, cp.weights, axis=0)
        return sp.array([xall, yall, zall]), allweights

    def plot_weights(self, translate=sp.array([0.0, 0.0, 0.0])):
        """ plot a mlab.mesh object colored by the weights """
        xt, yt, zt = translate[0], translate[1], translate[2]
        mlab.mesh(self.alldir[0] + xt, 
                  self.alldir[1] + yt, 
                  self.alldir[2] + zt, scalars=self.allweights)

#---------------------------------------------------------------------------------------
# DS class - responsible for direction, weight, and orbit info for conjugate directions 
#---------------------------------------------------------------------------------------                     
class DS(Sphere):
    """ The class directions will be a glorified ndarray with shape (3,)
    Additional functionality will make it useful for integration and 
    interpolation.  These objects will be elements in the Directions
    object. """
    def __init__(self, name, options, g, weight, vdir):
        """ Take directions and normalize them """
        super(DS, self).__init__(name, options)
        self.weight = sp.array(weight)
        self.vdir = vdir / (sp.sqrt(sp.inner(vdir, vdir)))
        #Define the self dictionary of directions
        self.orbit = sp.zeros([3, g.order])
        for i, act in enumerate(g):
            inv_act = g.inverse(act)
            self.orbit[:, i] = self.acted_on_by(inv_act).vdir
            self[act] = self.orbit[:, i]

    def plot_orbit(self, translate=[0.0, 0.0, 0.0], **options):
        """ Plot the orbit of a single direction """
        xt, yt, zt = translate[0], translate[1], translate[2]
        mlab.points3d(self.orbit[0] + xt, 
                      self.orbit[1] + yt, 
                      self.orbit[2] + zt,  **options)


#---------------------------------------------------------------------------------------
# Boundary - responsible for boundary sources of light
#---------------------------------------------------------------------------------------                     
class Boundary(Sphere):
    """ This class must get incoming values for a particular direction of transit
    and plot these values on a plane tangent to the unit sphere.  Plot vectors pointing 
    into the unit sphere to indicate where the beams have been defined.  This plot will
    be useful for testing that actions are correct """
    def __init__(self, name, options, grid, group):
        """ initialize a Boundary object """
        super(Boundary, self).__init__(name, options)

        # Define a local group variable
        self.g = group

        # Define the standard incoming position progection         
        unif = sp.array([grid.pos[1, 0, :, :], grid.pos[2, 0, :, :]])
        self.mask = unif[0]**2 + unif[1]**2 < 1 - grid.grid_stride[2] / 2
        # define the standard tangent plane 
        self.tang = sp.array([grid.x[0, :, :], grid.y[0, :, :], grid.z[0, :, :]])
        # define the xaxis projection onto the sphere 
        t = sp.real(sp.where(self.mask, sp.sqrt(1 - unif[0]**2 - unif[1]**2), 0))
        self.proj = sp.where(self.mask, 
                             sp.array([-t, unif[0], unif[1]]), 0)
        # set the initialized ds object 
        self.ds = None

        # set incoming beam data
        self.beam_type = options.beam_type
        self.p = options.p
        self.pc = options.pc
        self.d = options.d
        self.dc = options.dc

    # Update the stored ds during instantiation and at the start of each loop
    def set_new_ds(self, ds):
        """ Set a new direction element ds """
        self.ds = ds
        self.proj_dict = self.get_gspace_dict(self.proj)
        self.tang_dict = self.get_gspace_dict(self.tang)
        self.income_dict, self.minval, self.maxval = self.get_income_dict()

    # Get a dictionary for looking up points by the action that mapped theminto place. 
    def get_gspace_dict(self, points):
        """ take a 'points' array of shape (3, N, N), and rotate it back to its 
        place in standard 3D space. """
        # Set the rotation matrix
        r_mat = self.get_rotation_mat()
        # Locally rotated points
        self.vdir = sp.tensordot(r_mat, points, axes=([1, 0]))
        dict = {}
        for act in self.g:
            inv_act = self.g.inverse(act)
            dict[act] = self.acted_on_by(inv_act).vdir
        return dict

    def get_rotation_mat(self):
        """ Get the unitary rotation matrix that puts (1, 0, 0) in the given direction
        vdir."""
        x, y, z = self.ds.vdir[0], self.ds.vdir[1], self.ds.vdir[2]
        s = sp.sqrt(1.0-z**2)
        return sp.array([[x, y, z], [-y/s, x/s, 0.0], [-x*z/s, -y*z/s, s]]).transpose()

    # get incoming values 
    def get_income_dict(self):
        """ 
        Use the beam parameters, double henyey mixin, and the locally defined 
        direction and positions on the sphere to evaluate the incoming double henyey 
        sum function. 
        """
        # loop over all actions to define the incoming values
        income_dict = {} # initialize the dictionary
        minval, maxval = [], []
        for act in self.g:
            mu_p = sp.tensordot(self.p, self.proj_dict[act], axes=[1,0])
            mu_d = sp.tensordot(self.d, self.ds[act], axes=[1,0])
            #define the values with get_source_values
            values = sp.where(self.mask, 
                              get_source_values(mu_p, self.pc, 
                                                mu_d, self.dc, 
                                                self.beam_type), 0)
            minval.append(values.min())
            maxval.append(values.max())
            income_dict[act] = values
        minval = min(minval)
        maxval = max(maxval)
        return income_dict, minval, maxval


    # Plotting functions 
    def plot_orbit(self):
        """ plot incoming boundary light on tangent planes """
        for i, act in enumerate(self.g):
            m = i % 8
            n = (i - m) / 8
            shift = sp.array([[[5 * n]], [[5 * m]], [[0]]])
            #define projection, tangent plane, and values for this action
            proj = self.proj_dict[act] + shift
            tang = self.tang_dict[act] + shift
            xtang, ytang, ztang = tang[0], tang[1], tang[2]
            income = self.income_dict[act]
            supp = income >= 0
            # Plot points for the tangent planes and the projections
            self.plot_income_vectors(translate=shift[:,0,0])
            mlab.points3d(proj[0, supp], proj[1, supp], proj[2, supp], 
                          scale_factor = 0.7 / income.shape[0], 
                          color=(1, 1, 1))  # income[supp]) 
            mlab.mesh(xtang, ytang, ztang, scalars=income, 
                      transparent=True, vmin=self.minval, vmax=self.maxval)
            # plot direction direction of trasit, ds[act]
            x, y, z = -self.ds[act][0], -self.ds[act][1], -self.ds[act][2]
            u, v, w = self.ds[act][0], self.ds[act][1], self.ds[act][2]
            o = sp.ones((2,))
            x, y, z, u, v, w = o*x, o*y, o*z, o*u, o*v, o*w
            xt, yt, zt = shift[0, 0, 0], shift[1, 0, 0], shift[2, 0, 0]
            mlab.quiver3d(x + xt, y + yt, z + zt, u, v, w)

            
        mlab.colorbar(orientation='vertical')
        mlab.title('boundary.income_dict')


    def plot_income_vectors(self, translate=[0.0, 0.0, 0.0]):
        """ plot quivers to indicate incoming beams position and direction """
        xt, yt, zt = translate[0], translate[1], translate[2], 
        p = self.p + translate[sp.newaxis,:]
        # mlab.points3d(self.p[:,0], self.p[:,1], self.p[:,2],scale_factor=.1, color=(1,0,0))
        mlab.quiver3d(self.p[:,0]+xt, self.p[:,1]+yt, self.p[:,2]+zt, 
                      self.d[:,0], self.d[:,1], self.d[:,2], 
                      mode='cone', scale_factor=.2, color=(1,0,0))

#---------------------------------------------------------------------------------------
# S2Integrator Class
#---------------------------------------------------------------------------------------                     

#
# Unittest
#____________________________ 
if __name__ == '__main__':
    
    # get unit test module
    import unittest
    import group
    import domain 
    import options
    # define parameters
    N, M = 15, 4
    o = options.Options('o', N, M)
    g = group.Group(o)
    a = AllDirs('alldirs', o, g)
    ds = a.ds_list[3]
    grid = domain.Grid('grid', o)
    b = Boundary('b', o, grid, g)
    b.set_new_ds(ds)
    vdir = a.vdir

    # plot orbid
    mlab.figure(1)
    mlab.clf()
    b.plot_orbit()

    # Define test for options
    class TestS2Integrator(unittest.TestCase):
        """ Unittest suite for ensuring proper integration """
        def setUp(self):
            """ Prepare group and the directions vdir """
            self.g = g
            self.vdir = vdir
            self.ad = a
            self.b = b
        # Member shapes 
        def test_shape(self):
            """ Testing shapes of members """
            self.assertEqual(self.ad.weights_mask.shape, (self.ad.M, self.ad.M))
            self.assertEqual(self.ad.weights.shape, (self.ad.M, self.ad.M))
        # Normalized weights 
        def test_get_weights(self):
            """ Testing that weights sum to one """
            self.assertAlmostEqual(self.ad.allweights.sum(), 1.0, 1) # equal to four decimal places
        # test vector integration
        def test_loop(self):
            """ run a test loop and sum vector weights """
            self.sum = sp.zeros((3,))
            for ds in self.ad.ds_list:
                for act in self.g:
                    self.sum += ds[act] * ds.weight
                    self.b.set_new_ds(ds)
            print 'sum = ', self.sum
            self.assertAlmostEqual(abs(self.sum).sum(), 
                                   0, 2)
            

        # Integrate spherical harmonics exactly 
#         def test_integrate():
#             harm_integrals = []
#             # loop over all harmonic integrals to ensure they are accurate to 4 decimals
#             for index, true_harm in enumerate(harm_integrals):
# #                self.assertAlmostEqual(self.s.integrate(harm_FUNORVALS), true_harm, 4)
            


    # Run Unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(TestS2Integrator)
    unittest.TextTestRunner(verbosity=2).run(suite)
