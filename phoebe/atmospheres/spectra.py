"""
Interpolate and manipulate spectra.
"""
import os
import logging
import numpy as np
try:
    import pyfits
except ImportError:
    print("Soft warning: pyfits could not be found on your system, you can only use black body atmospheres and Gaussian synthetic spectral profiles")
from phoebe.algorithms import interp_nDgrid
from phoebe.utils import decorators

logger = logging.getLogger('ATMO.SP')

_aliases = {'atlas':'ATLASp_z0.0t0.0_a0.00_spectra.fits',\
            'bstar':'BSTAR2006_z0.000v2_vis_spectra.fits'}

@decorators.memoized
def _prepare_grid_spectrum(wrange,gridfile):
    """
    Example: 4570 - 4572 (N=200)
    """
    with pyfits.open(gridfile) as ff:
        grid_pars = np.zeros((2,len(ff)-1))
        grid_data = np.zeros((len(wrange)*2,len(ff)-1))
        for i,ext in enumerate(ff[1:]):
            grid_pars[0,i] = np.log10(ext.header['teff'])
            grid_pars[1,i] = ext.header['logg']
            wave = ext.data.field('wavelength')
            flux = ext.data.field('flux')
            cont = np.log10(ext.data.field('cont'))
            index1,index2 = np.searchsorted(wave,[wrange[0],wrange[-1]])
            #print index1,index2,wave.min(),wave.max(),len(wave),len(flux),len(cont)wrange[0],wrange[-1]
            index1 -= 1
            index2 += 1
            flux = np.interp(wrange,wave[index1:index2],flux[index1:index2])
            cont = np.interp(wrange,wave[index1:index2],cont[index1:index2])
            grid_data[:,i] = np.hstack([flux,cont])
            
    #-- create the pixeltype grid
    axis_values, pixelgrid = interp_nDgrid.create_pixeltypegrid(grid_pars,grid_data)
    
    return axis_values,pixelgrid        
            
            

def interp_spectable(gridfile,teff,logg,wrange):
    """
    Placeholder.
    
    Return line shape and continuum arrays.
    """
    #-- get the FITS-file containing the tables
    #-- retrieve structured information on the grid (memoized)
    if not os.path.isfile(gridfile):
        if gridfile in _aliases:
            gridfile = _aliases[gridfile]
        gridfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'tables','spectra',gridfile)
    axis_values,pixelgrid = _prepare_grid_spectrum(wrange,gridfile)
    #-- prepare input:
    values = np.zeros((2,len(teff)))
    values[0] = np.log10(teff)
    values[1] = logg
    N = len(wrange)
    try:
        pars = interp_nDgrid.interpolate(values,axis_values,pixelgrid)
    except IndexError:
        #pl.figure()
        #pl.title('Error in interpolation')
        #pl.plot(10**values[0],values[1],'ko')
        #pl.axhline(y=axis_values[1].min(),color='r',lw=2)
        #pl.axhline(y=axis_values[1].max(),color='r',lw=2)
        #pl.axvline(x=10**axis_values[0].min(),color='r',lw=2)
        #pl.axvline(x=10**axis_values[0].max(),color='r',lw=2)
        #pl.xlabel('Axis values teff')
        #pl.ylabel('Axis values logg')
        #print("DEBUG INFO: {}".format(axis_values))
        #pl.show()
        raise IndexError('Outside of grid: {:.1f}<teff<{:.1f}, {:.2f}<logg<{:.2f}'.format(10**values[0].min(),10**values[0].max(),values[1].min(),values[1].max()))
    pars[N:] = 10**pars[N:]
    pars = pars.reshape((2,len(wrange),-1))
    return pars