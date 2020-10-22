#!/usr/bin/env python
# coding: utf-8

# In[11]:


# import of standard python libraries
from __future__ import print_function
import numpy as np
import os
import time
import corner
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgetss

from mpl_toolkits.axes_grid1 import make_axes_locatable

# get_ipython().magic('matplotlib inline')
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
# plt.rc('text', usetex=True)
import sympy as sp 
from astropy.cosmology import FlatLambdaCDM

# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


# In[61]:


sigma_bkg = .05  #  background noise per pixel (Gaussian)
exp_time = 100.  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 100  #  cutout pixel size
deltaPix = 0.05  #  pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.1  # full width half max of PSF (only valid when psf_type='gaussian')
psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'
kernel_size = 91

z_lens = 0.5
z_source = 1.5

phi_ext = -0.5
gamma_ext = 0.06

theta_E = 1.66
gamma_lens = 1.98

center_x = 0.
center_y = 0.

center_x_lens_light = 0.
center_y_lens_light = 0.

source_x = 0.
source_y = 0.1

phi_source, q_source = 0.1, 0.8
phi_lens_light, q_lens_light = 0.9, 0.9

amp_source = 4000.
R_sersic_source = 0.2
n_sersic_source = 1.

amp_lens = 8000
R_sersic_lens = 0.4
n_sersic_lens = 2.

amp_ps=1000.
supersampling_factor = 1

v_min = -4
v_max = 2

e1_lens = 0.05
e2_lens = 0.05

def generate_image(sigma_bkg = sigma_bkg, exp_time = exp_time,
                   numPix = numPix, deltaPix = deltaPix, fwhm = fwhm, psf_type = psf_type, kernel_size = kernel_size,
                  z_source = z_source, z_lens = z_lens,
                  phi_ext = phi_ext, gamma_ext = gamma_ext,
                  theta_E = theta_E, gamma_lens = gamma_lens,
                  e1_lens = e1_lens, e2_lens = e2_lens,
                  center_x_lens_light = center_x_lens_light, center_y_lens_light = center_y_lens_light,
                  source_x = source_y, source_y = source_y,
                  q_source = q_source, phi_source = phi_source,
                  center_x = center_x, center_y = center_y,
                  amp_source = amp_source, R_sersic_source = R_sersic_source, n_sersic_source = n_sersic_source,
                  phi_lens_light = phi_lens_light, q_lens_light = q_lens_light,
                  amp_lens = amp_lens, R_sersic_lens = R_sersic_lens, n_sersic_lens = n_sersic_lens,
                  amp_ps = amp_ps,
                  supersampling_factor =supersampling_factor,
                  v_min = v_min, v_max = v_max,
                  lens_pos_eq_lens_light_pos = True):
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, 
                                             exp_time, sigma_bkg)
    data_class = ImageData(**kwargs_data)

    kwargs_psf = {'psf_type': psf_type,
                  'pixel_size': deltaPix, 'fwhm': fwhm}

    psf_class = PSF(**kwargs_psf)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)
    
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi=phi_ext, gamma=gamma_ext)
    kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2} 
    
    if lens_pos_eq_lens_light_pos:
        center_x = center_x_lens_light
        center_y = center_y_lens_light
    
   
    kwargs_spemd = {'theta_E': theta_E, 'gamma': gamma_lens,
                'center_x': center_x, 'center_y': center_y,
                'e1': e1_lens, 'e2': e2_lens} 
    lens_model_list = ['SPEP', 'SHEAR']
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list, z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    
    e1_source, e2_source = param_util.phi_q2_ellipticity(phi_source, q_source)
    
    kwargs_sersic_source = {'amp': amp_source, 'R_sersic': R_sersic_source, 'n_sersic': n_sersic_source,
                        'e1': e1_source, 'e2': e2_source, 
                        'center_x': source_x, 'center_y': source_y}
    
    source_model_list = ['SERSIC_ELLIPSE']
    kwargs_source = [kwargs_sersic_source]
    source_model_class = LightModel(light_model_list=source_model_list)
    ##
    e1_lens_light, e2_lens_light = param_util.phi_q2_ellipticity(phi_lens_light, q_lens_light)
    kwargs_sersic_lens = {'amp': amp_lens, 'R_sersic': R_sersic_lens, 'n_sersic': n_sersic_lens, 
                      'e1': e1_lens_light, 'e2': e2_lens_light, 
                      'center_x': center_x_lens_light, 'center_y': center_y_lens_light}
    lens_light_model_list = ['SERSIC_ELLIPSE']
    kwargs_lens_light = [kwargs_sersic_lens]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    ##
    lensEquationSolver = LensEquationSolver(lens_model_class)
    
    x_image, y_image = lensEquationSolver.findBrightImage(source_x, source_y,               #position of ps
                                                      kwargs_lens,                      #lens proporties
                                                      numImages=4,                      #expected number of images
                                                      min_distance=deltaPix,            #'resolution'
                                                      search_window=numPix * deltaPix)  #search window limits
    mag = lens_model_class.magnification(x_image, y_image,    #for found above ps positions
                                     kwargs=kwargs_lens)  # and same lens properties
    
    kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                           'point_amp': np.abs(mag)*amp_ps}] 
    point_source_list = ['LENSED_POSITION']
    point_source_class = PointSource(point_source_type_list=point_source_list, 
                                 fixed_magnification_list=[False])
    kwargs_numerics = {'supersampling_factor': supersampling_factor}
    imageModel = ImageModel(data_class,               # take generated above data specs
                        psf_class,                # same for psf
                        lens_model_class,         # lens model (gal+ext)
                        source_model_class,       # sourse light model
                        lens_light_model_class,   # lens light model
                        point_source_class,       # add generated ps images
                        kwargs_numerics=kwargs_numerics)
    image_sim = imageModel.image(kwargs_lens, 
                             kwargs_source,
                             kwargs_lens_light,
                             kwargs_ps)
    
    
    poisson = image_util.add_poisson(image_sim,
                                 exp_time=exp_time)
    bkg = image_util.add_background(image_sim,
                                sigma_bkd=sigma_bkg)
    
    image_sim = image_sim + bkg + poisson
    
    data_class.update_data(image_sim)
    kwargs_data['image_data'] = image_sim
    
    cmap_string = 'gray'
    cmap = plt.get_cmap(cmap_string)
    cmap.set_bad(color='b', alpha=1.)
    cmap.set_under('k')
    f, axes = plt.subplots(1, 1, figsize=(6, 6), 
                       sharex=False, sharey=False)
    ax = axes
    im = ax.matshow(np.log10(image_sim), origin='lower', 
                vmin=v_min, vmax=v_max, cmap=cmap, 
                extent=[0, 1, 0, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.autoscale(False)
    plt.show()
    return f


# In[65]:


# generate_image()


# In[63]:


# x_image


# In[64]:


#will be used later for fitting
# kwargs_model = {'lens_model_list': lens_model_list, 
#                  'lens_light_model_list': lens_light_model_list,
#                  'source_light_model_list': source_model_list,
#                 'point_source_model_list': point_source_list
#                  }


# In[ ]:





# In[ ]:




