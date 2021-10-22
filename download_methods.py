import os, gc, sys, timeit

import numpy as np
import math
from scipy.spatial.distance import cdist
from astropy import units as u
from mpi4py import MPI
import h5py

import eagle_IO.eagle_IO as E
import flares

norm = np.linalg.norm


def ndix_unique(x):

    """
    From "https://stackoverflow.com/questions/54734545/indices-of-unique-values-in-n-dimensional-array?noredirect=1&lq=1"

    Returns an N-dimensional array of indices
    of the unique values in x
    ----------
    x: np.array
       Array with 1 dimension
    Returns
    -------
    - 1D-array of sorted unique values
    - Array of arrays. Each array contains the indices where a
      given value in x is found
    """

    ix = np.argsort(x)
    u, ix_u = np.unique(x[ix], return_index=True)
    ix_ndim = np.unravel_index(ix, x.shape)
    ix_ndim = np.c_[ix_ndim] if x.ndim > 1 else ix
    return u, np.split(ix_ndim, ix_u[1:])


def extract_info(num, tag, inp='FLARES'):

    """
    Returns set of pre-defined properties of galaxies from a
    region in FLARES `num` for `tag`. Selects only galaxies
    with more than 100 star+gas particles inside 30pkpc
    ----------
    Args:
        num : str
            the FLARES/G-EAGLE id of the sim; eg: '00', '01', ...
        tag : str
            the file tag; eg: '000_z015p00', '001_z014p000',...., '011_z004p770'

    """

    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print (F"Extracing information from {inp} {num} {tag} (rank: {rank}, size: {size})")

    if inp == 'FLARES':
        sim_type = 'FLARES'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        num = str(num)
        if len(num) == 1:
            num =  '0'+num
        _dir = fl.directory
        sim = F"{_dir}GEAGLE_{num}/data/"

    elif inp == 'REF':
        sim_type = 'PERIODIC'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        sim = fl.ref_directory

    elif inp == 'AGNdT9':
        sim_type = 'PERIODIC'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        sim = fl.agn_directory

    else:
        ValueError("Type of input simulation not recognized")

    if rank == 0:
        print (F"Sim location: {sim}, tag: {tag}")

    #Simulation redshift
    z = E.read_header('SUBFIND', sim, tag, 'Redshift')
    a = E.read_header('SUBFIND', sim, tag, 'ExpansionFactor')
    boxl = E.read_header('SUBFIND', sim, tag, 'BoxSize')/E.read_header('SUBFIND', sim, tag, 'HubbleParam')

    ####### Galaxy global properties  #######
    # SubhaloMass = E.read_array('SUBFIND', sim, tag, '/Subhalo/Mass', numThreads=4, noH=True, physicalUnits=True)
    Maperture = E.read_array('SUBFIND', sim, tag, '/Subhalo/ApertureMeasurements/Mass/030kpc', numThreads=4, noH=True, physicalUnits=True)
    mstar = Maperture[:,4]
    sgrpno = E.read_array('SUBFIND', sim, tag, '/Subhalo/SubGroupNumber', numThreads=4)
    grpno = E.read_array('SUBFIND', sim, tag, '/Subhalo/GroupNumber', numThreads=4)


    if inp == 'FLARES':
        ## Selecting the subhalos within our region
        cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', noH=False, physicalUnits=False, numThreads=4) #units of cMpc/h
        cen, r, min_dist = fl.spherical_region(sim, tag)  #units of cMpc/h
        indices = np.where((mstar*1e10 >= 10**7.) & (norm(cop-cen, axis=1)<=fl.radius))[0]

    else:
        indices = np.where(mstar*1e10 >= 10**7.)[0]

    cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', noH=True, physicalUnits=True, numThreads=4)
    # sfr_inst =  E.read_array('SUBFIND', sim, tag, '/Subhalo/ApertureMeasurements/SFR/030kpc', numThreads=4, noH=True, physicalUnits=True)


    ####### Particle properties #######

    #dm particle
    dm_cood = E.read_array('PARTDATA', sim, tag, '/PartType1/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    dm_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType1/SubGroupNumber', numThreads=4)
    dm_grpn = E.read_array('PARTDATA', sim, tag, '/PartType1/GroupNumber', numThreads=4)

    #Gas particle
    gp_cood = E.read_array('PARTDATA', sim, tag, '/PartType0/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    gp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType0/SubGroupNumber', numThreads=4)
    gp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType0/GroupNumber', numThreads=4)
    gp_sfr = E.read_array('PARTDATA', sim, tag, '/PartType0/StarFormationRate', noH=True, physicalUnits=True, numThreads=4)

    #Star particle
    try:
        sp_cood = E.read_array('PARTDATA', sim, tag, '/PartType4/Coordinates', noH=True, physicalUnits=True, numThreads=4)
        sp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType4/SubGroupNumber', numThreads=4)
        sp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType4/GroupNumber', numThreads=4)
        SP = True
    except:
        SP = False
        print("No star particles found")


    #Black hole particle
    """
    Subgrid properties are the ones required.
    Only at high masses are the subhalo and particle properties trace each other
    """
    try:
        bh_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType5/SubGroupNumber', numThreads=4)
        bh_grpn = E.read_array('PARTDATA', sim, tag, '/PartType5/GroupNumber', numThreads=4)
        bh_mass = E.read_array('PARTDATA', sim, tag, '/PartType5/BH_Mass', noH=True, physicalUnits=True, numThreads=4)
        bh_cood = E.read_array('PARTDATA', sim, tag, '/PartType5/Coordinates', numThreads=4, noH=True, physicalUnits=True)
        BH = True
    except:
        BH = False
        print("No Black hole particles found")




    ###########################  For identifying spurious galaxies and remerging them to the parent  ###########################

    #First method: just using the EAGLE method of merging them
    #to the nearby subhalo
    #Second method: use Will's criterion in MEGA to merge only
    #particles of those group that are identified as single
    #entities ----- not done for now

    #Identifying the index of the spurious array within the
    #array `indices`

    spurious_indices = np.where((Maperture[:,0][indices] == 0) | (Maperture[:,1][indices] == 0) | (Maperture[:,4][indices] == 0))[0]
    if len(spurious_indices)>0:
        #Calculating the distance of the spurious to the other subhalos
        dist_to_others = cdist(cop[indices[spurious_indices]], cop[indices])

        #To take into account the fact that the spurious subhalos
        #themselves as well as others are present within
        #`indices` at the moment
        dist_to_others[:, spurious_indices] = np.nan

        #Parent is classified as the nearest subhalo to the spurious
        parent = indices[np.nanargmin(dist_to_others, axis=1)]

        #returns the index of the parent and its associated spurious
        #as an array of arrays. `spurious_of_parent` is linked to
        #the `spurious` which is defined below so you can get the
        #original index back wrt to the whole dataset
        parent, spurious_of_parent = ndix_unique(parent)

        #remove the spurious from indices so they aren't counted twice
        #in the subhalo/particle property collection, but retain
        #information (`spurious` array) on where they are within the
        #whole dataset for later use
        spurious = indices[spurious_indices]
        indices = np.delete(indices, spurious_indices)

        del spurious_indices, dist_to_others
        sp_ok = True

    else:
        sp_ok = False

    gc.collect()


    comm.Barrier()

    part = int(len(indices)/size)
    num_subhalos = int(len(sgrpno))

    if rank == 0:
        if inp != 'FLARES': num = ''
        print("Extracting required properties for {} subhalos from {} region {} at z = {} of boxsize = {}".format(len(indices), inp, num, z, boxl))

    #For getting black hole subgrid masses
    tbhindex = np.zeros(num_subhalos, dtype = np.int32)
    tbh_cood = np.zeros((num_subhalos, 3), dtype = np.float64)
    tbh_mass = np.zeros(num_subhalos, dtype = np.float32)

    tsindex = np.zeros(num_subhalos, dtype = np.int32)


    if inp == 'FLARES':
        if rank!=size-1:
            thisok = indices[rank*part:(rank+1)*part]
        else:
            thisok = indices[rank*part:]

    else:
        #Size needs to be a perfect cube to work
        l = boxl / (size)**(1/3)
        sz = (size)**(1/3)
        dl = 10.
        xyz = np.zeros((size,8,3))
        count=0
        for xx in range(int(sz)):
            for yy in range(int(sz)):
                for zz in range(int(sz)):
                    xyz[count] = np.array([[xx, yy, zz], [xx+1, yy, zz], [xx, yy+1, zz], [xx, yy, zz+1], [xx+1, yy+1, zz], [xx+1, yy, zz+1], [xx, yy+1, zz+1], [xx+1, yy+1, zz+1]])
                    count+=1


        this_xyz = xyz[rank]*l
        max_xyz = np.max(this_xyz, axis=0)
        min_xyz = np.min(this_xyz, axis=0)


        thisok = np.ones(len(indices), dtype=bool)
        for xx in range(3):
            thisok*=np.logical_and(cop[indices][:,xx]/a>=min_xyz[xx], cop[indices][:,xx]/a<=max_xyz[xx])
        thisok = indices[thisok]
        # print (thisok, rank, max_xyz, min_xyz)
        # print (cop[thisok])

        #Dividing the gas particles into a cell for current task
        dd = np.ones(len(dm_cood), dtype=bool)
        for xx in range(3):
            dd*=np.logical_or((min_xyz[xx]-dl<=dm_cood[:,xx]/a)*(dm_cood[:,xx]/a<=max_xyz[xx]+dl), np.logical_or((min_xyz[xx]-dl<=dm_cood[:,xx]/a+boxl)*(dm_cood[:,xx]/a+boxl<=max_xyz[xx]+dl), (min_xyz[xx]-dl<=dm_cood[:,xx]/a-boxl)*(dm_cood[:,xx]/a-dl<=max_xyz[xx]+dl)))
        dd = np.where(dd)[0]

        dm_cood = dm_cood[dd]
        dm_sgrpn = dm_sgrpn[dd]
        dm_grpn = dm_grpn[dd]

        gg = np.ones(len(gp_cood), dtype=bool)
        for xx in range(3):
            gg*=np.logical_or((min_xyz[xx]-dl<=gp_cood[:,xx]/a)*(gp_cood[:,xx]/a<=max_xyz[xx]+dl), np.logical_or((min_xyz[xx]-dl<=gp_cood[:,xx]/a+boxl)*(gp_cood[:,xx]/a+boxl<=max_xyz[xx]+dl), (min_xyz[xx]-dl<=gp_cood[:,xx]/a-boxl)*(gp_cood[:,xx]/a-dl<=max_xyz[xx]+dl)))
        gg = np.where(gg)[0]

        gp_cood = gp_cood[gg]
        gp_sgrpn = gp_sgrpn[gg]
        gp_grpn = gp_grpn[gg]
        gp_sfr = gp_sfr[gg]

        #Dividing the star particles into a cell for current task
        if SP:
            ss = np.ones(len(sp_cood), dtype=bool)
            for xx in range(3):
                ss*=np.logical_or((min_xyz[xx]-dl<=sp_cood[:,xx]/a)*(sp_cood[:,xx]/a<=max_xyz[xx]+dl), np.logical_or((min_xyz[xx]/a-dl<=sp_cood[:,xx]/a+boxl)*(sp_cood[:,xx]/a+boxl<=max_xyz[xx]+dl), (min_xyz[xx]-dl<=sp_cood[:,xx]/a-boxl)*(sp_cood[:,xx]/a-boxl<=max_xyz[xx]+dl)))
            ss = np.where(ss)[0]

            sp_cood = sp_cood[ss]
            sp_sgrpn = sp_sgrpn[ss]
            sp_grpn = sp_grpn[ss]

        #Dividing the black hole particles into a cell for current task
        if BH:
            bb = np.ones(len(bh_cood), dtype=bool)
            for xx in range(3):
                bb*=np.logical_or((min_xyz[xx]-dl<=bh_cood[:,xx]/a)*(bh_cood[:,xx]/a<=max_xyz[xx]+dl), np.logical_or((min_xyz[xx]-dl<=bh_cood[:,xx]/a+boxl)*(bh_cood[:,xx]/a+boxl<=max_xyz[xx]+dl), (min_xyz[xx]-dl<=bh_cood[:,xx]/a-boxl)*(bh_cood[:,xx]/a-boxl<=max_xyz[xx]+dl)))
            bb = np.where(bb)[0]

            bh_sgrpn = bh_sgrpn[bb]
            bh_grpn = bh_grpn[bb]
            bh_mass = bh_mass[bb]
            bh_cood = bh_cood[bb]

    gc.collect()

    tdnum = np.zeros(len(thisok)+1, dtype = np.int32)
    tsnum = np.zeros(len(thisok)+1, dtype = np.int32)
    tgnum = np.zeros(len(thisok)+1, dtype = np.int32)
    ind = np.array([])

    tdindex = np.zeros(len(dm_grpn), dtype = np.int32)
    tgindex = np.zeros(len(gp_grpn), dtype = np.int32)

    if SP:
        tsindex = np.zeros(len(sp_grpn), dtype = np.int32)

    gc.collect()

    kk = 0
    # dist = 0.1 #in pMpc for 100 pkpc Aperture, writes out particle properties within this aperture
    sel_dist = 0.03 #in pMpc for 30 pkpc Aperture, only galaxies with more than 100 star + gas particles within this aperture is written out to the master file. Only spurious galaxies within 30 pkpc are selected
    bounds = np.array([boxl, boxl, boxl])   #https://stackoverflow.com/a/11109244
    for ii, jj in enumerate(thisok):

        #start = timeit.default_timer()

        d_ok = np.where((dm_sgrpn-sgrpno[jj]==0) & (dm_grpn-grpno[jj]==0))[0]

        g_ok = np.where((gp_sgrpn-sgrpno[jj]==0) & (gp_grpn-grpno[jj]==0))[0]
        tmp = gp_cood[g_ok]-cop[jj]
        if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
        g_ok_sel = g_ok[norm(tmp,axis=1)<=sel_dist]

        if SP:
            s_ok = np.where((sp_sgrpn-sgrpno[jj]==0) & (sp_grpn-grpno[jj]==0))[0]
            tmp = sp_cood[s_ok]-cop[jj]
            if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
            s_ok_sel = s_ok[norm(tmp,axis=1)<=sel_dist]
        else:
            s_ok = np.array([])

        if BH:
            bh_ok = np.where((bh_sgrpn-sgrpno[jj]==0) & (bh_grpn-grpno[jj]==0))[0]

        if sp_ok:
            if jj in parent:
                this_spurious = np.where(parent == jj)[0]

                for _jj in spurious[spurious_of_parent[this_spurious[0]]]:

                    #To apply Will's recombine method, it should
                    #be applied here, instead of the next block

                    spurious_d_ok = np.where((dm_sgrpn-sgrpno[_jj]==0) & (dm_grpn-grpno[_jj]==0))[0]
                    tmp = dm_cood[spurious_d_ok]-cop[jj]
                    if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
                    d_ok = np.append(d_ok, spurious_d_ok[norm(tmp,axis=1)<=sel_dist])

                    spurious_g_ok = np.where((gp_sgrpn-sgrpno[_jj]==0) & (gp_grpn-grpno[_jj]==0))[0]
                    tmp = gp_cood[spurious_g_ok]-cop[jj]
                    if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
                    g_ok = np.append(g_ok, spurious_g_ok[norm(tmp,axis=1)<=sel_dist])
                    g_ok_sel = np.append(g_ok_sel, spurious_g_ok[norm(tmp,axis=1)<=sel_dist])

                    if SP:
                        spurious_s_ok = np.where((sp_sgrpn-sgrpno[_jj]==0) & (sp_grpn-grpno[_jj]==0))[0]
                        tmp = sp_cood[spurious_s_ok]-cop[jj]
                        if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
                        s_ok = np.append(s_ok, spurious_s_ok[norm(tmp,axis=1)<=sel_dist])
                        s_ok_sel = np.append(s_ok_sel, spurious_s_ok[norm(tmp,axis=1)<=sel_dist])

                    if BH:
                        spurious_bh_ok = np.where((bh_sgrpn-sgrpno[_jj]==0) & (bh_grpn-grpno[_jj]==0))[0]
                        tmp = bh_cood[spurious_bh_ok]-cop[jj]
                        if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
                        bh_ok = np.append(bh_ok, spurious_bh_ok[norm(tmp,axis=1)<=sel_dist])

                #Add in here the subhalo properties that needed
                #to be added due to spurious
                # tsfr_inst_spurious[jj] = np.sum(gp_sfr[g_ok])
                # tmstar_spurious[jj] = np.sum(mstar[spurious[spurious_of_parent[this_spurious[0]]]])
                # tSubhaloMass_spurious[jj] = np.sum(SubhaloMass[spurious[spurious_of_parent[this_spurious[0]]]])


        #stop = timeit.default_timer()

        if len(s_ok_sel) + len(g_ok_sel) >= 100:

            #print ("Calculating indices took {}s".format(np.round(stop - start,6)))
            # start = timeit.default_timer()

            #Extracting subgrid black hole properties
            if BH:
                if len(bh_ok>0):

                    tbh_max_index = np.argmax(bh_mass[bh_ok])
                    tbh_mass[jj] = bh_mass[bh_ok[tbh_max_index]]

                    if inp=='FLARES':
                        tbhindex[jj] = bh_ok[tbh_max_index]
                    else:
                        tbhindex[jj] = bb[bh_ok[tbh_max_index]]

            if SP:
                tsnum[kk+1] = len(s_ok)
                scum = np.cumsum(tsnum)
                sbeg = scum[kk]
                send = scum[kk+1]
                if inp=='FLARES':
                    tsindex[sbeg:send] = s_ok
                else:
                    tsindex[sbeg:send] = ss[s_ok]


            tdnum[kk+1] = len(d_ok)
            tgnum[kk+1] = len(g_ok)
            dcum = np.cumsum(tdnum)
            gcum = np.cumsum(tgnum)
            dbeg = dcum[kk]
            dend = dcum[kk+1]
            gbeg = gcum[kk]
            gend = gcum[kk+1]

            if inp=='FLARES':
                tdindex[dbeg:dend] = d_ok
                tgindex[gbeg:gend] = g_ok
            else:
                tdindex[dbeg:dend] = dd[d_ok]
                tgindex[gbeg:gend] = gg[g_ok]

            # stop = timeit.default_timer()
            # print ("Assigning arrays took {}s".format(np.round(stop - start,6)))
            gc.collect()

            kk+=1

        else:

            ind = np.append(ind, ii)

    ##End of loop ii, jj##

    del dm_sgrpn, dm_grpn, dm_cood, gp_sgrpn, gp_grpn, gp_cood, gp_sfr
    if SP: del sp_sgrpn, sp_grpn, sp_cood,
    if BH: del bh_sgrpn, bh_grpn, bh_mass

    gc.collect()


    thisok = np.delete(thisok, ind.astype(int))
    tbhindex = tbhindex[thisok]
    tbh_mass = tbh_mass[thisok]

    tdtot = np.sum(tdnum)
    tstot = np.sum(tsnum)
    tgtot = np.sum(tgnum)

    tdnum = tdnum[1:len(thisok)+1]
    tsnum = tsnum[1:len(thisok)+1]
    tgnum = tgnum[1:len(thisok)+1]

    tdindex = tdindex[:tdtot]
    tsindex = tsindex[:tstot]
    tgindex = tgindex[:tgtot]

    comm.Barrier()

    gc.collect()


    if rank == 0:
        print ("Gathering data from different processes")

    indices = comm.gather(thisok, root=0)


    bhindex = comm.gather(tbhindex, root=0)
    bh_mass = comm.gather(tbh_mass, root=0)

    del thisok, tbhindex, tbh_mass
    gc.collect()

    dnum = comm.gather(tdnum, root=0)
    del tdnum
    snum = comm.gather(tsnum, root=0)
    del tsnum
    gnum = comm.gather(tgnum, root=0)
    del tgnum

    dindex = comm.gather(tdindex, root=0)
    del tdindex
    sindex = comm.gather(tsindex, root=0)
    del tsindex
    gindex = comm.gather(tgindex, root=0)
    del tgindex

    gc.collect()

    ok_centrals = 0.

    if rank == 0:

        print ("Gathering completed")

        indices = np.concatenate(np.array(indices))
        dindex = np.concatenate(np.array(dindex))
        sindex = np.concatenate(np.array(sindex))
        gindex = np.concatenate(np.array(gindex))
        bhindex = np.concatenate(np.array(bhindex))

        bh_mass = np.concatenate(np.array(bh_mass))

        dnum = np.concatenate(np.array(dnum))
        snum = np.concatenate(np.array(snum))
        gnum = np.concatenate(np.array(gnum))

        ok_centrals = grpno[indices] - 1

        cop = cop[indices]/a
        sgrpno = sgrpno[indices]
        grpno = grpno[indices]


    return ok_centrals, indices, sgrpno, grpno, cop, dnum, snum, gnum, dindex, sindex, gindex, bhindex, bh_mass
##End of function `extract_info`


def save_to_hdf5(num, tag, dset, name, desc, dtype = None, unit = '', group = 'Galaxy', inp='FLARES', data_folder = 'data/', verbose = False, overwrite=False):

    if dtype is None:
        dtype = dset.dtype

    num = str(num)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num
        filename = './{}/FLARES_{}_sp_info.hdf5'.format(data_folder, num)
        sim_type = 'FLARES'


    elif inp == 'REF' or inp == 'AGNdT9':
        filename = F"./{data_folder}/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'


    else:
        ValueError("Type of input simulation not recognized")


    if verbose: print("Writing out required properties to hdf5")

    fl = flares.flares(fname = filename,sim_type = sim_type)
    fl.create_group(tag)

    if verbose: print("Creating necessary groups")
    groups = group.split(os.sep)
    ogrp = ''
    for grp in groups:
        ogrp += os.sep + grp
        fl.create_group(f'{tag}/{ogrp}')

    if unit is not None:
        fl.create_dataset(dset, name, f'{tag}/{group}', dtype = dtype, desc = desc, overwrite=overwrite)
    else:
        fl.create_dataset(dset, name, f'{tag}/{group}', dtype = dtype, desc = desc, overwrite=overwrite)




def recalculate_derived_subhalo_properties(inp, num, tag, S_len, G_len, D_len, \
                                           S_index, G_index, D_index):
    """
    Recalculate subhalo properties, such as the stellar/total mass and SFR,
    after inclusion of spurious galaxies.
    """

    if inp == 'FLARES':
        sim_type = 'FLARES'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        num = str(num)
        if len(num) == 1:
            num =  '0'+num
        _dir = fl.directory
        sim = F"{_dir}GEAGLE_{num}/data/"

    elif inp == 'REF':
        sim_type = 'PERIODIC'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        sim = fl.ref_directory

    elif inp == 'AGNdT9':
        sim_type = 'PERIODIC'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        sim = fl.agn_directory

    else:
        ValueError("Type of input simulation not recognized")


    # gp_sfr = E.read_array('PARTDATA', sim, tag, '/PartType0/StarFormationRate', noH=True, physicalUnits=True, numThreads=1)
    try:
        gp_mass = E.read_array('PARTDATA', sim, tag, '/PartType0/Mass',
                               noH=True, physicalUnits=True, numThreads=1)
    except:
        gp_mass = np.array([])

    try:
        sp_mass = E.read_array('PARTDATA', sim, tag, '/PartType4/Mass',
                               noH=True, physicalUnits=True, numThreads=1)
    except:
        sp_mass = np.array([])

    try:
        dm_pmass = E.read_header('PARTDATA',sim,tag,'MassTable')[1] /\
                   E.read_header('PARTDATA',sim,tag,'HubbleParam')
    except:
        dm_pmasss = np.array([])

    sbegin = np.zeros(len(S_len), dtype = np.int64)
    send = np.zeros(len(S_len), dtype = np.int64)
    sbegin[1:] = np.cumsum(S_len)[:-1]
    send = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int64)
    gend = np.zeros(len(G_len), dtype = np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)

    SMass = np.zeros(len(S_len))
    GMass = np.zeros(len(G_len))
    # total_SFR = np.zeros(len(S_len))

    for jj in range(len(sbegin)):
        SMass[jj] = np.sum(sp_mass[S_index[sbegin[jj]:send[jj]]])
        GMass[jj] = np.sum(gp_mass[G_index[gbegin[jj]:gend[jj]]])
        # total_SFR[jj] = np.sum(gp_sfr[G_index[gbegin[jj]:gend[jj]]])

    DMass = D_len * dm_pmass

    return SMass, GMass, DMass # , total_SFR



def get_recent_SFR(num, tag, t = 100, aperture_size = 30, inp = 'FLARES'):
    '''
    Calculate and save the star formation rate averaged over different timescales and aperture sizes. Also outputs the stellar mass averaged over different aperture sizes.

    :num: region number
    :tag: snapshot tag (str)
    :t: timescale over which to average over (Myr, float / list)
    :aperture_size: aperture (centred on the centre of potential of the subhalo) over which to calculate the SFR (kpc, float / list)
    :inp: simulation type (str)

    :return: flares master file with requested properties added
    '''

    if not isinstance(t,list): t = [t]
    if not isinstance(aperture_size,list): aperture_size = [aperture_size]

    if inp == 'FLARES':
        sim_type = inp
        num = "%02d"%int(num)
        sim = F"./data/FLARES_{num}_sp_info.hdf5"
    elif (inp == 'REF') or (inp == 'AGNdT9'):
        sim = F"./data/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
    else:
        ValueError(F"No input option of {inp}")


    z = float(tag[5:].replace('p','.'))
    a = 1. / (1+z)

    with h5py.File(sim, 'r') as hf:
        S_len = np.array(hf[F'{tag}/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len = np.array(hf[F'{tag}/Galaxy'].get('G_Length'), dtype = np.int64)
        COP = np.array(hf[F'{tag}/Galaxy'].get('COP'), dtype = np.float64) * a

        # S_mass = np.array(hf[F'{tag}/Particle'].get('S_MassInitial'), dtype = np.float64)
        S_mass = np.array(hf[F'{tag}/Particle'].get('S_Mass'), dtype = np.float64)
        S_age = np.array(hf[F'{tag}/Particle'].get('S_Age'), dtype = np.float64)*1e3 #Age is in Gyr, so converting the array to Myr
        S_coods = np.array(hf[F'{tag}/Particle'].get('S_Coordinates'), dtype = np.float64) * a


    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    SFR = {_ap: {_t: np.zeros(len(begin)) for _t in t} for _ap in aperture_size}
    Mstar = {_ap: np.zeros(len(begin)) for _ap in aperture_size}

    for jj, kk in enumerate(begin):

        if (begin[jj] - end[jj]) == 0: ## no particles
            continue

        this_age = S_age[begin[jj]:end[jj]]
        this_cood = S_coods[:,begin[jj]:end[jj]]
        this_mass = S_mass[begin[jj]:end[jj]]

        ## filter by age and aperture
        for _ap in aperture_size:
            aperture_mask = (cdist(this_cood.T, np.array([COP[:,jj]])) < _ap * 1e-3)[:,0]

            if np.sum(aperture_mask) > 0:
                Mstar[_ap][jj] = np.sum(this_mass[aperture_mask])

            for _t in t:
                age_mask = this_age <= _t
                ok = np.where(aperture_mask & age_mask)[0]

                if len(ok) > 0:
                    SFR[_ap][_t][jj] = np.sum(this_mass[ok])/(_t*1e6) * 1e10

    return SFR, Mstar


def get_aperture_inst_SFR(num, tag, aperture_size = 30, inp = 'FLARES'):

    if not isinstance(aperture_size,list): aperture_size = [aperture_size]

    if inp == 'FLARES':
        sim_type = inp
        num = "%02d"%int(num)
        sim = F"./data/FLARES_{num}_sp_info.hdf5"
    elif (inp == 'REF') or (inp == 'AGNdT9'):
        sim = F"./data/EAGLE_{inp}_sp_info.hdf5"
        sim_type = 'PERIODIC'
    else:
        ValueError(F"No input option of {inp}")


    z = float(tag[5:].replace('p','.'))
    a = 1. / (1+z)

    with h5py.File(sim, 'r') as hf:
        COP = np.array(hf[F'{tag}/Galaxy'].get('COP'), dtype = np.float64) * a
        G_len = np.array(hf[F'{tag}/Galaxy'].get('G_Length'), dtype = np.int64)
        G_coods = np.array(hf[F'{tag}/Particle'].get('G_Coordinates'), dtype = np.float64) * a
        G_SFR = np.array(hf[F'{tag}/Particle'].get('G_SFR'), dtype = np.float64)


    ## Instantaneous SFR
    begin = np.zeros(len(G_len), dtype = np.int64)
    end = np.zeros(len(G_len), dtype = np.int64)
    begin[1:] = np.cumsum(G_len)[:-1]
    end = np.cumsum(G_len)

    inst_SFR = {_ap: np.zeros(len(begin)) for _ap in aperture_size}

    for jj, kk in enumerate(begin):
        this_cood = G_coods[:,begin[jj]:end[jj]]
        this_sfr = G_SFR[begin[jj]:end[jj]]

        for _ap in aperture_size:
            aperture_mask = (cdist(this_cood.T, np.array([COP[:,jj]])) < _ap * 1e-3)[:,0]

            if np.sum(aperture_mask) > 0:
                inst_SFR[_ap][jj] = np.sum(this_sfr[aperture_mask])


    return inst_SFR


    # fl = flares(sim, sim_type)
    # fl.create_dataset(SFR, F"{tag}/Galaxy/SFR/SFR_{t}",
    #                   desc = F"SFR of the galaxy averaged over the last {t}Myr", unit = "Msun/yr")

    # if save_mass:
    #     fl.create_dataset(SFR, F"{tag}/Galaxy/Mstar/Mstar_{t}",
    #                       desc = F"Stellar mass of the galaxy within a {aperture_size} kpc aperture", unit = "Msun")

    # print (F"Saved the SFR averaged over {t}Myr with tag {tag} for {inp} to file")




if __name__ == "__main__":

    ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    num = str(ii)
    tag = str(tag)
    inp = str(inp)
    data_folder = str(data_folder)

    num = "%02d"%int(num)

    # save_to_hdf5(num, tag, inp=inp, data_folder=data_folder)

    ok_centrals, indices, sgrpno, grpno, cop, dnum, snum, gnum, dindex, sindex, gindex, bhindex, bh_mass = extract_info(num, tag, inp)

    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print ("#################    Saving required properties to hdf5     ###################")

    if rank != 0:
        del ok_centrals, indices, sgrpno, grpno, cop, dnum, snum, gnum, dindex, sindex, gindex, bhindex, bh_mass

    gc.collect()

    if rank == 0:
        save_to_hdf5(num, tag, indices, 'Indices', 'Index of the galaxy in the resimulation', dtype='int64', group='Galaxy', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, ok_centrals, 'Central_Indices', 'Index of the central galaxies in the resimulation', dtype='int64', group='Galaxy', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, dindex, 'DM_Index', 'Index of selected dark matter particles in the particle data', dtype='int64', group='Particle', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, sindex, 'S_Index', 'Index of selected star particles in the particle data', dtype='int64', group='Particle', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, gindex, 'G_Index', 'Index of selected gas particles in the particle data', dtype='int64', group='Particle', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, bhindex, 'BH_Index', 'Index of selected black hole particles in the particle data', dtype='int64', group='Particle', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, grpno, 'GroupNumber', 'Group number of the galaxy', dtype='int64', group='Galaxy', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, sgrpno, 'SubGroupNumber', 'Subgroup number of the galaxy', dtype='int64', group='Galaxy', inp=inp, overwrite=True)

        save_to_hdf5(num, tag, dnum, 'DM_Length', 'Number of dark matter particles', dtype='int64', group='Galaxy', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, snum, 'S_Length', 'Number of star particles', dtype='int64', group='Galaxy', inp=inp, overwrite=True)
        save_to_hdf5(num, tag, gnum, 'G_Length', 'Number of gas particles', dtype='int64', group='Galaxy', inp=inp, overwrite=True)

        save_to_hdf5(num, tag, cop.T, 'COP', desc = 'Number of gas particles', group='Galaxy', inp=inp, unit='cMpc', overwrite=True)
        save_to_hdf5(num, tag, bh_mass, 'BH_Mass', desc = 'Mass of the most massive black hole in the subgroup', group='Galaxy', inp=inp, unit='1e10 Msun', overwrite=True)
