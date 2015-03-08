### ====================================================================
###  Python-file
###     author:          William G. K. Martin
###     filename:        interp.py
### ====================================================================


import scipy as sp
from scipy.linalg import norm
from enthought.mayavi import mlab


#parameterize a bunch of circles
n = 100 
t = sp.linspace(0, 2*sp.pi, n)

a = sp.array([(sp.cos(s), sp.sin(s), 0.) for s in t] + 
             [(sp.cos(s), 0., sp.sin(s)) for s in t]).transpose()
b = sp.array([(0., sp.cos(s), sp.sin(s)) for s in t]).transpose()

c = sp.array([(sp.sin(s), 0., 0.) for s in t] +
             [(0., sp.sin(s), 0.) for s in t] + 
             [(0., 0., sp.sin(s)) for s in t]).transpose()


def plot_ball_axes(center, label):
    """ plot axes and title for a ball"""
    
    # assertions 
    assert center.shape == (3,), "expected center to be a ndarray with shape (3,)"
    assert type(label) == type('string'), "label must be of type string"
    
    # shift axes to correct center
    center = center[:, sp.newaxis]
    a_loc = a + center
    b_loc = b + center
    c_loc = c + center

    # plot the axes
    mlab.plot3d(a_loc[0], a_loc[1], a_loc[2], tube_radius=.01, opacity=.25)
    mlab.plot3d(b_loc[0], b_loc[1], b_loc[2], tube_radius=.01, opacity=.25)

    mlab.plot3d(c_loc[0], c_loc[1], c_loc[2], tube_radius=.01, opacity=.25)
    
    # plot label
    mlab.text(float(center[0]), 
              float(center[1]), 
              label, 
              z=float(center[2])+1.02,
              width=.1)


# Plot a list of values at the given nodes in space
#________________________________________________________________________________
def plot_grid_list(x, y, z, value_list, label_list, fig=None):
    """ 
    plot all the elements in value list, according to their size (as either a
    vector or scalar field) at the points x, y, z and with the corresponding 
    title in label list
    """
    plot_ratio = 1./100.
    

    # check vargin
    for val in value_list:
        if val.shape[0] == 3:
            assert val.shape == (3,) + x.shape, "unexpected value matrix shape"
        else:
            assert val.shape == x.shape, "unexpected value matrix shape"
            
    # Setup figure scene
    if fig:
        mlab.figure(fig)
        mlab.clf()
    else:
        mlab.figure()

    # loop over and plot values 
    center = sp.array([0., 0., 0.])
    
    for i, val in enumerate(value_list):
        # use quiver3d for vectors
        if val.shape[0]==3:
            val_norm = sp.sqrt(val[0]**2 + val[1]**2 + val[2]**2)
            tol = val_norm/val_norm.max() >= plot_ratio
            

            if tol.any():
                mlab.quiver3d(x[tol] + center[0], 
                              y[tol] + center[1], 
                              z[tol] + center[2], 
                              val[0, tol], val[1, tol], val[2, tol], 
                              mode='arrow')
            plot_ball_axes(center, label_list[i])
        
        # plot using points3d
        else:
            tol = abs(val) / abs(val).max() >= plot_ratio
            if tol.any():
                mlab.points3d(x[tol] + center[0], 
                              y[tol] + center[1], 
                              z[tol] + center[2], val[tol])
            
            plot_ball_axes(center, label_list[i])
            
        # shift center for next plot
        center[1] += 3.
            
            




# def plot_ball_axes(centers=[sp.zeros((3,))], labels = ['no_label_given']):
#     """draw axes for unit balls around the given centers"""
        
#     # check incoming  
#     assert len(centers) == len(labels), "centers and labels must have same length"
#     assert type(centers) == type([]), "expected centers to be type list, but got type " + repr(type(centers))
#     assert type(labels) ==  type([]), "expected labels to be type list, but got type " + repr(type(centers))

#     for i, center in enumerate(centers)
    

#     while centers:
#         # new centers
#         center = centers.pop()
#         label = labels.pop()


#         a = a + center
        



if __name__=='__main__':
    
    plot_ball_axes(sp.array([3, 2., 4.]), 'title for axes plot')
