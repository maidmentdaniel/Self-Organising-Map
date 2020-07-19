"""
Daniel Maidment

Tue May 28 12:57:01 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


"""##################################################################"""
plt.style.use('seaborn-paper')


def config_axis(ax = None,
                x_lim = None, X_0 = None,
                y_lim = None, Y_0 = None,
                grd = True, minorgrd = False,
                mult_x = 0.2, mult_y = 0.2,
                Eng = True):
    if(X_0 != None):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(mult_x*X_0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator((mult_x/5)*X_0))
    if(Eng):
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
        ax.yaxis.set_major_formatter(ticker.EngFormatter())
    if(Y_0 != None):
        ax.yaxis.set_major_locator(ticker.MultipleLocator(mult_y*Y_0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(mult_y/5*Y_0))
    if(grd == True):
        ax.grid(b = True, which = 'major', axis = 'both')
    else:
        ax.grid(b = False, which = 'major', axis = 'both')
    if(minorgrd == True):
        ax.grid(b = True, which = 'minor', axis = 'both')
    else:
        ax.grid(b = False, which = 'minor', axis = 'both')

    if(x_lim != None):
        ax.set_xlim(x_lim)
    if(y_lim != None):
        ax.set_ylim(y_lim)
    return ax
"""##################################################################"""

import SelfOrganisingMap as som

dataFile = som.readData('iris_data.txt')

testSOM = som.selfOrganisingMap( gShape=(30, 30),
                                lam = 5000, rate_mx = 1, 
                                data = dataFile, fx_seed = True,
                                debug = True)

testSOM.train()

som_map = testSOM.SOM

U_mat = som.calc_euclid_matrix(som_map)

fig, ax  = plt.subplots(1,1, figsize = (10,10))
ax.imshow(U_mat, cmap = 'copper')
plt.show()
#
display_map = testSOM.visualise_data(n = 3, debug = True)

fig, ax  = plt.subplots(1,1, figsize = (10, 10))
col_map = plt.cm.get_cmap( 'terrain', 4)
im = ax.imshow(display_map, cmap=col_map)
fig.colorbar(im, ax = ax)
plt.show()



