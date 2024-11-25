import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def _isolate_neck(coords_all, teffs_all, cutoff = 0.,component=1, plot=False):
    if component == 1:
        cond = coords_all[:,0] >= 0+cutoff
    elif component == 2:
        cond = coords_all[:,0] <= 1-cutoff
    else:
        raise ValueError
        
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        axes[0].scatter(coords_all[cond][:,0], coords_all[cond][:,1])
        axes[1].scatter(coords_all[cond][:,0], teffs_all[cond])
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('teff')
        fig.tight_layout()
        plt.show()
     
    return coords_all[cond], teffs_all[cond], np.argwhere(cond).flatten()


def _dist(p1, p2):
    (x1, y1, z1), (x2, y2, z2) = p1, p2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def _isolate_sigma_fitting_regions(coords_neck, teffs_neck, direction='x', cutoff=0., component=1, plot=False):

    distances = [_dist(p1, p2) for p1, p2 in combinations(coords_neck, 2)]
    min_dist = np.min(distances[distances != 0])
    
    if direction == 'x':
        cond = (coords_neck[:,1] >= -0.2*min_dist) & (coords_neck[:,1] <= 0.2*min_dist) 
    
    elif direction == 'y':
        if component == 1:
            cond = coords_neck[:,0] <= 0+cutoff+0.15*min_dist
        elif component == 2:
            cond = coords_neck[:,0] >= 1-cutoff-0.15*min_dist
        else:
            raise ValueError
    else:
        raise ValueError
        
    if plot:
        if direction == 'x':
            plt.scatter(coords_neck[cond][:,0], teffs_neck[cond])
            plt.show()
        elif direction == 'y':
            plt.scatter(coords_neck[cond][:,1], teffs_neck[cond])
            plt.show()

    return coords_neck[cond], teffs_neck[cond]


def _compute_new_teff_at_neck(coords1, teffs1, coords2, teffs2, w=0.5, offset=0.):
    
    distances1 = [_dist(p1, p2) for p1, p2 in combinations(coords1, 2)]
    min_dist1 = np.min(distances1[distances1 != 0])
    distances2 = [_dist(p1, p2) for p1, p2 in combinations(coords2, 2)]
    min_dist2 = np.min(distances2[distances2 != 0])
    
    x_neck = np.average((coords1[:,0].max(), coords2[:,0].min()))
    amplitude_1 = teffs1.max()
    amplitude_2 = teffs2.max()
    
    teffs_neck1 = teffs1[coords1[:,0] >= coords1[:,0].max() - 0.25*min_dist1]
    teffs_neck2 = teffs2[coords2[:,0] <= coords2[:,0].min() + 0.25*min_dist2]
    
    teff1 = np.max(teffs2)#np.average(teffs_neck1)
    teff2 = np.average(teffs_neck2)
    tavg = w*teff1 + (1-w)*teff2
    if tavg > teffs2.max():
        print('Warning: Tavg > Teff2, setting new temperature to 1 percent of Teff2 max. %i > %i' % (int(tavg), int(teffs2.max())))
        tavg = teffs2.max() - 0.01*teffs2.max()
    
    return x_neck, tavg
    

def _compute_sigmax(Tavg, x, x0, offset, amplitude):
    return ((-1)*(x-x0)**2/np.log((Tavg-offset)/amplitude))**0.5
    

def _fit_sigma_amplitude(coords, teffs, offset=0., cutoff=0., direction='y', component=1, plot=False):
    
    def gaussian_1d(x, s):
        a = 1./s**2
        g = offset + amplitude * np.exp(- (a * ((x - x0) ** 2)))
        return g

    from scipy.optimize import curve_fit
    
    if direction == 'y':
        coord_ind = 1
        x0 = 0.
    elif direction == 'x':
        coord_ind = 0
        if component == 1:
            x0 = 0+cutoff
        elif component == 2:
            x0 = 1-cutoff
        else:
            raise ValueError
    else:
        raise ValueError
    
    amplitude = teffs.max() - offset
    sigma_0 = 0.5
    result = curve_fit(gaussian_1d, 
              xdata=coords[:,coord_ind], 
              ydata=teffs, 
              p0=(sigma_0,),
              bounds=[0.01,1000])
    
    sigma = result[0]
    model = gaussian_1d(coords[:,coord_ind], sigma)
    
    if plot:
        plt.scatter(coords[:,coord_ind], teffs)
        plt.scatter(coords[:,coord_ind], model)
        plt.show()
    
    return sigma, amplitude, model
    

def _compute_twoD_Gaussian(coords, sigma_x, sigma_y, amplitude, cutoff=0., offset=0., component=1):
    y0 = 0.
    x0 = 0+cutoff if component == 1 else 1-cutoff
    a = 1. / sigma_x ** 2
    b = 1. / sigma_y ** 2
    return offset + amplitude * np.exp(- (a * ((coords[:, 0] - x0) ** 2))) * np.exp(- (b * ((coords[:, 1] - y0) ** 2)))


def gaussian_smoothing(xyz1, teffs1, xyz2, teffs2, w=0.5, cutoff=0., offset=0.):
    coords_neck_1, teffs_neck_1, cond1 = _isolate_neck(xyz1, teffs1, cutoff = cutoff, component=1, plot=False)
    coords_neck_2, teffs_neck_2, cond2 = _isolate_neck(xyz2, teffs2, cutoff = cutoff, component=2, plot=False)

    x_neck, Tavg = _compute_new_teff_at_neck(coords_neck_1, teffs_neck_1, coords_neck_2, teffs_neck_2, w=w, offset=offset)

    sigma_x1 = _compute_sigmax(Tavg, x_neck, x0=0+cutoff, offset=offset, amplitude=teffs_neck_1.max())
    sigma_x2 = _compute_sigmax(Tavg, x_neck, x0=1-cutoff, offset=offset, amplitude=teffs_neck_2.max())

    coords_fit_y1, teffs_fit_y1 = _isolate_sigma_fitting_regions(coords_neck_1, teffs_neck_1, direction='y', cutoff=cutoff, 
                                                                        component=1, plot=False)
    coords_fit_y2, teffs_fit_y2 = _isolate_sigma_fitting_regions(coords_neck_2, teffs_neck_2, direction='y', cutoff=cutoff,
                                                                        component=2, plot=False)

    sigma_y1, amplitude_y1, model_y1 = _fit_sigma_amplitude(coords_fit_y1, teffs_fit_y1, offset, direction='y', 
                                                                        component=1, plot=False)
    sigma_y2, amplitude_y2, model_y2 = _fit_sigma_amplitude(coords_fit_y2, teffs_fit_y2, offset, direction='y', 
                                                                        component=2, plot=False)

    new_teffs1 =  _compute_twoD_Gaussian(coords_neck_1, sigma_x1, sigma_y1, teffs_neck_1.max(), 
                                                cutoff=cutoff, offset=offset, component=1)
    new_teffs2 = _compute_twoD_Gaussian(coords_neck_2, sigma_x2, sigma_y2, teffs_neck_2.max(), 
                                                cutoff=cutoff, offset=offset, component=2)

    # print(cond1, len(cond1), len(new_teffs1))
    # print(cond2, len(cond2), len(new_teffs2))
    teffs1[cond1] = new_teffs1
    teffs2[cond2] = new_teffs2

    return teffs1, teffs2


def lateral_transfer(t2s, teffs2, mixing_power, teff_ratio):
    """
    Scales the temperatures of the secondary to that of the primary only in a horizontal band the size of the contact's
    neck. This implies mixing occurs due to mass transfer across the neck
    """
    x2s = t2s[:,0]
    y2s = t2s[:,1]
    z2s = t2s[:,2]

    # do latitude dependence
    z2s_neck = z2s[x2s < 1]
    y2s_neck = y2s[x2s < 1]
    lat = np.min(np.sqrt(z2s_neck ** 2 + y2s_neck ** 2))
    filt = (z2s>-lat) & (z2s<lat)  # select band extending the (projected) height of the neck
    bandfactor = (lat-np.abs(z2s[filt])) ** mixing_power
    latitude_dependence = bandfactor / bandfactor.max()

    # do longitude dependence
    phi = np.arctan2(y2s[filt], x2s[filt] - 1) + np.pi  #[0, 2pi]
    k = 3
    lambd = 2*np.pi
    longitude_dependence = np.exp(-(phi/lambd)**k)  # weibull to vary from 1 to 0 over some range phi>0

    # adjust teffs
    teffs2[filt] *= 1 + (1 - teff_ratio) * latitude_dependence * longitude_dependence
    return teffs2


def isotropic_transfer(t2s, teffs2, mixing_power, teff_ratio):
    """
    Scales the temperatures of the secondary to that of the primary, parametrized by the radial distance from the
    origin (which is the center of the primary)
    """
    d2s = np.sqrt(t2s[:,0]*t2s[:,0] + t2s[:,1]*t2s[:,1] + t2s[:,2]*t2s[:,2])
    teffs2 *= 1 + (1-teff_ratio) * (1 - ((d2s - d2s.min()) / (d2s.max() - d2s.min()))) ** mixing_power
    return teffs2


def perfect_transfer(teff2s, teff_ratio):
    """
    Scales the temperatures of the secondary to that of the primary, implying perfect thermal mixing occurred deep in
    the interior of the stars.
    """
    teff2s *= 1/teff_ratio
    return teff2s


def spotty_transfer(t2s, teffs2):
    """
    Scales the temperatures of the secondary by randomly placing 10 spots on its surface which are a little hotter.
    Implies mixing occurs due to increases magnetism on the secondary.
    """
    d2s = np.sqrt(t2s[:,0]*t2s[:,0] + t2s[:,1]*t2s[:,1] + t2s[:,2]*t2s[:,2])
    for s in range(10):
        idx = int(len(d2s)*np.random.rand())
        size = 0.3*np.random.rand()
        factor = 1.1-0.025*np.random.rand()

        ds = np.sqrt((t2s[:,0]-t2s[idx,0])**2 + (t2s[:,1]-t2s[idx,1])**2 + (t2s[:,2]-t2s[idx,2])**2)
        teffs2[ds<size] *= factor

    return teffs2


def smooth_teffs(xyz1, teffs1, xyz2, teffs2, mixing_method='lateral', mixing_power=0.5, teff_ratio=1.):
    if mixing_method == 'lateral':
        teffs2 = lateral_transfer(xyz2, teffs2, mixing_power, teff_ratio)
    elif mixing_method == 'isotropic':
        teffs2 = isotropic_transfer(xyz2, teffs2, mixing_power, teff_ratio)
    elif mixing_method == 'spotty':
        teffs2 = spotty_transfer(xyz2, teffs2)
    elif mixing_method == 'perfect':
        teffs2 = perfect_transfer(teffs2, teff_ratio)
    else:
        teffs1, teffs2 = gaussian_smoothing(xyz1, teffs1, xyz2, teffs2)
    return teffs1, teffs2