from calc_Hlos import new_cal_HLOS
import numpy as np

from astropy import units as u
from astropy import constants as c
conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)

# test to see whether the hydrogen column density function works

# let's set up a scenario with one stellar particle and one gas particle

# redshift
z = 5

# stellar coordinates (convert co-moving to physical) [pMpc]
s_coords = np.array([[1600,1600,1600]])/(1+z)

# gas coordinates [pMpc]
g_coords = np.array([[1600,1600,2000],[1600,1600,1000]])/(1+z)

# gas particle mass [Msun]
# (in raw data it's 1E10 but this is removed in get_relevant_particles)
g_mass = np.array([10000000, 10000000])

# gas particle hydrogen mass frac
g_hfrac = np.array([0.75, 0.75])

# gas temperature [K]
g_temp = np.array([10000, 10000])

# gas smoothing length [pMpc]
g_sml = np.array([0.005, 0.005])

# kernel look-up table
kinp = np.load('./data/kernel_sph-anarchy.npz', allow_pickle=True)
lkernel = kinp['kernel']

# number of bins in look-up table
header = kinp['header']
kbins = header.item()['bins']

print('kernel file keys:', list(kinp.keys()))
print('kernel header:', header)
print('kernel:', lkernel)

# let's integrate the kernel argh idk htf to do this
# xx = np.linspace(0,1,num=10001)
# print(len(xx))
# print(len(lkernel))
# kernel_tot = np.trapz(lkernel, xx)
# print('kernel sum:', kernel_tot)

# get hydrogen column density [Msun/pc^2]
hcd = new_cal_HLOS(s_coords, g_coords, g_mass, g_hfrac, g_temp, g_sml,
                   lkernel, kbins)*conv

print('hydrogen column density:', hcd, '[Msun/pc^2]')

# get escape fraction
m_sun = u.solMass # [kg]
m_h = c.m_p # [kg] (ignore electrons)
hcd = hcd * (m_sun/m_h) # [no. H atoms / pc^2]
sigma = 6.3 * 10**-18  # photoionisation cross section [cm^2]
sigma = sigma * (u.cm**2).to(u.pc**2) # [pc^2]
fesc = np.exp(-sigma*hcd)

print('fesc:', fesc)




# let's see how fesc changes as we shift the gas particle
n = 20
shifted_coords = np.zeros((n,3))
shifted_coords[:,2] = 1605/(1+z) # z-coords
shifted_coords[:,1] = 1600/(1+z) # y-coords
gxs = np.linspace(1599.95, 1600.05, num=n)
shifted_coords[:,0] = gxs/(1+z) # let's shift the x-coords

fescs = []
for coord in shifted_coords:
    # add an extra gas particle
    coord = np.append([coord], [np.array([1600,1600,1000])/(1+z)], axis=0)
    
    # get hydrogen column density
    hcd = new_cal_HLOS(s_coords, coord, g_mass, g_hfrac, g_temp, g_sml,
                   lkernel, kbins)*conv

    # get escape fraction
    m_sun = u.solMass # [kg]
    m_h = c.m_p # [kg] (ignore electrons)
    hcd = hcd * (m_sun/m_h) # [no. H atoms / pc^2]
    sigma = 6.3 * 10**-18  # photoionisation cross section [cm^2]
    sigma = sigma * (u.cm**2).to(u.pc**2) # [pc^2]
    fesc = np.exp(-sigma*hcd)

    fescs.append(fesc)

print('for gas x-coords:', gxs, 'cMpc')
print('g_sml:', 0.005*(1+z), 'cMpc')
print('fescs:', fescs)

