### ====================================================================
###  Python-file
###     author:          William G. K. Martin
###     filename:        directions.py
###     version:         
###     created:         
###       on:            
###     last modified:   
###       at:            
###     URL:             
###     email:           
###  
### ====================================================================
#
### ====================================================================
#
# This file will create a class "Directions", with the following 
# attributes:
#
# 
### ====================================================================
# Attributes:
# cos1, cos2, cos3
# weight
# plotWeights
# M
# 
### ====================================================================


### ====================================================================
#   Import libraries and define classes
### ====================================================================
import scipy as sp
import scipy.integrate as spInt
import enthought.mayavi.mlab as mlab
import copy as copy

import group
import sphere_mixins as mix




### ====================================================================
#   Subclass nd array
### ====================================================================
class DS(sp.ndarray, mix.DSMixIn):
    """ The class directions will be a glorified ndarray with shape (3,)
    Additional functionality will make it useful for integration and 
    interpolation.  These objects will be elements in the Directions
    object. """
    def __init__(self, name, weight, dir):
        """ Take directions and normalize them """
        self.name = name 
        self.dir = sp.array(dir)
        





### ====================================================================
#   Testing script 
### ====================================================================

ds_test = {'print':True, 'plot':True}


# Test Direction Class
if ds_test['print']:
    ds = DS('area element', .1, (1.0, .2, .2))
    print "==============================================================="
    print ds.name, ' : test output '
    print "---------------------------------------------------------------"
    print 'ParGrid attributes (non-standard):'
    for att in dir(ds):
        if att not in dir(sp.ndarray):
            print '\t', '%-15s' %(att)
            #         print 'grid_shape is ', d.grid_shape
            #         print 'ray_shape is ', d.ray_shape
            #         print 'grid_stride is ', d.grid_stride
            #         print 'ray_stride is ', d.ray_stride
    print "---------------------------------------------------------------"







    
    
