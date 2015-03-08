# ====================================================================
#  Python-file
#     author:          William G. K. Martin
#     filename:        transport3d.py
# ====================================================================
"""
This script computes and visualizes solutions to the trasport equation
in three spatial dimentions.  The options class can be used to set
boundary sources, PDE parameters, etc. 

The method is an explicit descrete-ordinate method (as opposed to
monte-carlo).  The non-linear transport equation is solved in the unit
ball for successive orders of scattering.

"""

#   Import libraries and define classes
#___________________________________________________________________________

import scipy as sp
import enthought.mayavi.mlab as mlab

import sys


### ====================================================================
#   instantiate objects
### ====================================================================

# Options sub-objects
import options, group, domain, sphere
N = 13                    # grid parameter
M = 11                    # direction parameter
options = options.Options('options', N, M, group_name='group48')

# Group sub-objects
# import group
actions = group.Group(options)

# Domain sub-objects
# import domain
grid = domain.Grid('grid', options)
rays = domain.Rays('rays', options)

# Sphere sub-objects
# import sphere
bound = sphere.Boundary('bound', options, grid, actions) 
all_dirs = sphere.AllDirs('all_dirs', options, actions)

### ====================================================================
#   Solve transport using successive orders of scattering 
### ====================================================================

# Loop over orders of scattering until completion or failure 
while (grid.scat_order <= 3):
    
    grid.setup_scatter_loop()

    # Loop over the set of non-redundant directions
    for ds in all_dirs.ds_list:

        # Set new rays and grid ds
        rays.set_new_ds(ds)
        grid.set_new_ds(ds)

        # prepare a dictionary of incoming intensities
        if grid.scat_order == 0: 
            bound.set_new_ds(ds) 

        # iterate over all actions 
        for act in actions:
            
            # act on grid
            grid.set_new_act(act)
            
            # sample grid
            rays.sample_from(grid)
            
            # solve ode
            if grid.scat_order == 0:
                rays.solve_ballistic(bound.income_dict[act])
            else:
                rays.solve_scattering()
        
            # resample 
            rays.resample_grid_rad()

            # update loop_fluxes
            grid.update_loop_fluxes(rays.grid_rad)

    # finalize grid to update scat_order and check for convergence
    grid.store_loop_fluxes()
    print grid.scat_order

### ====================================================================
#   Visualize results
### ====================================================================
# actions.plot()
# bound.plot()
# all_dirs.plot()
# parg.plot()
# solg.plot()
#grid.plot_fluxes()
#grid.plot_param()
grid.plot_sos()

### ====================================================================
#   Testing
### ====================================================================
    
