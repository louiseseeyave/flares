import gc, sys, timeit

import numpy as np
import math
from scipy.spatial.distance import cdist
from astropy import units as u
from mpi4py import MPI

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

    print (F"Extracing information from {inp} {num} {tag}")

    if inp == 'FLARES':
        sim_type = 'FLARES'
        fl = flares.flares(fname = './data/',sim_type=sim_type)
        num = str(num)
        if len(num) == 1:
            num =  '0'+num
        dir = fl.directory
        sim = F"{dir}GEAGLE_{num}/data/"

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
    SubhaloMass = E.read_array('SUBFIND', sim, tag, '/Subhalo/Mass', numThreads=4, noH=True, physicalUnits=True)
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
    sfr_inst =  E.read_array('SUBFIND', sim, tag, '/Subhalo/ApertureMeasurements/SFR/030kpc', numThreads=4, noH=True, physicalUnits=True)

    ####### Particle properties #######

    #dm particle
    dm_cood = E.read_array('PARTDATA', sim, tag, '/PartType1/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    dm_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType1/SubGroupNumber', numThreads=4)
    dm_grpn = E.read_array('PARTDATA', sim, tag, '/PartType1/GroupNumber', numThreads=4)

    #Star particle
    sp_cood = E.read_array('PARTDATA', sim, tag, '/PartType4/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    sp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType4/SubGroupNumber', numThreads=4)
    sp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType4/GroupNumber', numThreads=4)

    #Gas paticle
    gp_cood = E.read_array('PARTDATA', sim, tag, '/PartType0/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    gp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType0/SubGroupNumber', numThreads=4)
    gp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType0/GroupNumber', numThreads=4)
    gp_sfr = E.read_array('PARTDATA', sim, tag, '/PartType0/StarFormationRate', noH=True, physicalUnits=True, numThreads=4)

    #Black hole particle
    #subgrid properties are theones required
    #Only for high masses the subhalo and particle properties trace each other
    try:
        bh_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType5/SubGroupNumber', numThreads=4)
        bh_grpn = E.read_array('PARTDATA', sim, tag, '/PartType5/GroupNumber', numThreads=4)
        bh_mass = E.read_array('PARTDATA', sim, tag, '/PartType5/BH_Mass', noH=True, physicalUnits=True, numThreads=4)
        bh_cood = E.read_array('PARTDATA', sim, tag, '/PartType5/Coordinates', numThreads=4, noH=True, physicalUnits=True)
        BH = True
    except:
        BH = False
        print ("No Black hole particles found")




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

    #Arrays that needs addition:
    tmstar_spurious = np.zeros(num_subhalos, dtype = np.float32)
    tsfr_inst_spurious = np.zeros(num_subhalos, dtype = np.float64)
    tSubhaloMass_spurious = np.zeros(num_subhalos, dtype = np.float32)


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

    dn = len(dm_grpn)
    sn = len(sp_grpn)
    gn = len(gp_grpn)

    tdindex = np.zeros(dn, dtype = np.int32)
    tsindex = np.zeros(sn, dtype = np.int32)
    tgindex = np.zeros(gn, dtype = np.int32)

    gc.collect()

    kk = 0
    dist = 0.03 #in pMpc for 30pkpc Aperture
    bounds = np.array([boxl, boxl, boxl])   #https://stackoverflow.com/a/11109244
    for ii, jj in enumerate(thisok):

        #start = timeit.default_timer()

        d_ok = np.where((dm_sgrpn-sgrpno[jj]==0) & (dm_grpn-grpno[jj]==0))[0]
        tmp = dm_cood[d_ok]-cop[jj]
        if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
        d_ok = d_ok[norm(tmp,axis=1)<=dist]

        s_ok = np.where((sp_sgrpn-sgrpno[jj]==0) & (sp_grpn-grpno[jj]==0))[0]
        tmp = sp_cood[s_ok]-cop[jj]
        if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
        s_ok = s_ok[norm(tmp,axis=1)<=dist]

        g_ok = np.where((gp_sgrpn-sgrpno[jj]==0) & (gp_grpn-grpno[jj]==0))[0]
        tmp = gp_cood[g_ok]-cop[jj]
        if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
        g_ok = g_ok[norm(tmp,axis=1)<=dist]
        if BH:
            bh_ok = np.where((bh_sgrpn-sgrpno[jj]==0) & (bh_grpn-grpno[jj]==0))[0]
            tmp = bh_cood[bh_ok]-cop[jj]
            if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
            bh_ok = bh_ok[norm(tmp,axis=1)<=dist]

        if sp_ok:
            if jj in parent:
                this_spurious = np.where(parent == jj)[0]

                for _jj in spurious[spurious_of_parent[this_spurious[0]]]:

                    #To apply Will's recombine method, it should
                    #be applied here, instead of the next block

                    spurious_d_ok = np.where((dm_sgrpn-sgrpno[_jj]==0) & (dm_grpn-grpno[_jj]==0))[0]
                    tmp = dm_cood[spurious_d_ok]-cop[jj]
                    if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
                    d_ok = np.append(d_ok, spurious_d_ok[norm(tmp,axis=1)<=dist])

                    spurious_s_ok = np.where((sp_sgrpn-sgrpno[_jj]==0) & (sp_grpn-grpno[_jj]==0))[0]
                    tmp = sp_cood[spurious_s_ok]-cop[jj]
                    if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
                    s_ok = np.append(s_ok, spurious_s_ok[norm(tmp,axis=1)<=dist])

                    spurious_g_ok = np.where((gp_sgrpn-sgrpno[_jj]==0) & (gp_grpn-grpno[_jj]==0))[0]
                    tmp = gp_cood[spurious_g_ok]-cop[jj]
                    if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
                    g_ok = np.append(g_ok, spurious_g_ok[norm(tmp,axis=1)<=dist])

                    if BH:
                        spurious_bh_ok = np.where((bh_sgrpn-sgrpno[_jj]==0) & (bh_grpn-grpno[_jj]==0))[0]
                        tmp = bh_cood[spurious_bh_ok]-cop[jj]
                        if inp!='FLARES': tmp = np.min(np.dstack(((tmp) % bounds, (-tmp) % bounds)), axis = 2)
                        bh_ok = np.append(bh_ok, spurious_bh_ok[norm(tmp,axis=1)<=dist])

                #Add in here the subhalo properties that needed
                #to be added due to spurious
                tsfr_inst_spurious[jj] = np.sum(gp_sfr[g_ok])
                tmstar_spurious[jj] = np.sum(mstar[spurious[spurious_of_parent[this_spurious[0]]]])
                tSubhaloMass_spurious[jj] = np.sum(SubhaloMass[spurious[spurious_of_parent[this_spurious[0]]]])


        #stop = timeit.default_timer()

        if len(s_ok) + len(g_ok) >= 100:

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

            tdnum[kk+1] = len(d_ok)
            tsnum[kk+1] = len(s_ok)
            tgnum[kk+1] = len(g_ok)

            dcum = np.cumsum(tdnum)
            scum = np.cumsum(tsnum)
            gcum = np.cumsum(tgnum)
            dbeg = dcum[kk]
            dend = dcum[kk+1]
            sbeg = scum[kk]
            send = scum[kk+1]
            gbeg = gcum[kk]
            gend = gcum[kk+1]

            if inp=='FLARES':
                tdindex[dbeg:dend] = d_ok
                tsindex[sbeg:send] = s_ok
                tgindex[gbeg:gend] = g_ok
            else:
                tdindex[dbeg:dend] = dd[d_ok]
                tsindex[sbeg:send] = ss[s_ok]
                tgindex[gbeg:gend] = gg[g_ok]

            # stop = timeit.default_timer()
            # print ("Assigning arrays took {}s".format(np.round(stop - start,6)))
            gc.collect()

            kk+=1

        else:

            ind = np.append(ind, ii)

    ##End of loop ii, jj##

    del dm_sgrpn, dm_grpn, dm_cood, sp_sgrpn, sp_grpn, sp_cood, gp_sgrpn, gp_grpn, gp_cood, gp_sfr
    if BH:
        del bh_sgrpn, bh_grpn, bh_mass

    gc.collect()


    thisok = np.delete(thisok, ind.astype(int))
    tbhindex = tbhindex[thisok]
    tbh_mass = tbh_mass[thisok]

    tmstar_spurious = tmstar_spurious[thisok]
    tsfr_inst_spurious = tsfr_inst_spurious[thisok]
    tSubhaloMass_spurious = tSubhaloMass_spurious[thisok]

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

    mstar_spurious = comm.gather(tmstar_spurious, root=0)
    sfr_inst_spurious = comm.gather(tsfr_inst_spurious, root=0)
    SubhaloMass_spurious = comm.gather(tSubhaloMass_spurious, root=0)

    del thisok, tbhindex, tbh_mass, tmstar_spurious, tsfr_inst_spurious, tSubhaloMass_spurious
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

        mstar_spurious = np.concatenate(np.array(mstar_spurious))
        sfr_inst_spurious = np.concatenate(np.array(sfr_inst_spurious))
        SubhaloMass_spurious = np.concatenate(np.array(SubhaloMass_spurious))

        dnum = np.concatenate(np.array(dnum))
        snum = np.concatenate(np.array(snum))
        gnum = np.concatenate(np.array(gnum))


        ok_centrals = grpno[indices] - 1

        SubhaloMass = SubhaloMass[indices]+SubhaloMass_spurious
        mstar = mstar[indices]+mstar_spurious
        sfr_inst = sfr_inst[indices]+sfr_inst_spurious
        cop = cop[indices]/a
        sgrpno = sgrpno[indices]
        grpno = grpno[indices]


    return ok_centrals, indices, sgrpno, grpno, cop, SubhaloMass, mstar, sfr_inst, dnum, snum, gnum, dindex, sindex, gindex, bhindex, bh_mass
##End of function `extract_info`

def save_to_hdf5(num, tag, inp='FLARES', data_folder = 'data/'):

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



    ok_centrals, indices, sgrpno, grpno, cop, SubhaloMass, mstar, sfr_inst, dnum, snum, gnum, dindex, sindex, gindex, bhindex, bh_mass = extract_info(num, tag, inp)


    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        print ("#################    Saving required properties to hdf5     #############")
        print ("#################   Number of processors being used is {}   #############".format(size))
        print (F"Writing to {filename}")

    if rank != 0:
        del ok_centrals, indices, sgrpno, grpno, cop, SubhaloMass, mstar, sfr_inst, dnum, snum, gnum, dindex, sindex, gindex, bhindex, bh_mass

    gc.collect()

    if rank == 0:

        print("Wrting out required properties to hdf5")

        fl = flares.flares(fname = filename,sim_type = sim_type)
        fl.create_group(tag)
        if inp == 'FLARES':
            dir = fl.directory
            sim = F"{dir}GEAGLE_{num}/data/"

        elif inp == 'REF':
            sim = fl.ref_directory

        elif inp == 'AGNdT9':
            sim = fl.agn_directory


        fl.create_group('{}/Galaxy'.format(tag))
        fl.create_group('{}/Particle'.format(tag))


        fl.create_dataset(indices, 'Indices', '{}/Galaxy'.format(tag), dtype = 'int64',
            desc = 'Index of the galaxy in the resimulation')
        fl.create_dataset(ok_centrals, 'Central_Indices', '{}/Galaxy'.format(tag), dtype = 'int64',
            desc = 'Index of the central galaxies in the resimulation')

        fl.create_dataset(dindex, 'DM_Index', '{}/Particle'.format(tag), dtype = 'int64',
            desc = 'Index of selected DM particles in the particle data')
        fl.create_dataset(sindex, 'S_Index', '{}/Particle'.format(tag), dtype = 'int64',
            desc = 'Index of selected star particles in the particle data')
        fl.create_dataset(gindex, 'G_Index', '{}/Particle'.format(tag), dtype = 'int64',
            desc = 'Index of selected gas particles in the particle data')
        fl.create_dataset(bhindex, 'BH_Index', '{}/Particle'.format(tag), dtype = 'int64',
            desc = 'Index of selected black hole particles in the particle data')


        fl.create_dataset(grpno, 'GroupNumber', '{}/Galaxy'.format(tag), dtype = 'int64',
            desc = 'Group Number of the galaxy')
        fl.create_dataset(sgrpno, 'SubGroupNumber', '{}/Galaxy'.format(tag), dtype = 'int64',
            desc = 'Subgroup Number of the galaxy')

        fl.create_dataset(dnum, 'DM_Length', '{}/Galaxy'.format(tag), dtype = 'int64',
            desc = 'Number of DM particles inside 30pkpc')
        fl.create_dataset(snum, 'S_Length', '{}/Galaxy'.format(tag), dtype = 'int64',
            desc = 'Number of star particles inside 30pkpc')
        fl.create_dataset(gnum, 'G_Length', '{}/Galaxy'.format(tag), dtype = 'int64',
            desc = 'Number of gas particles inside 30pkpc')


        fl.create_dataset(SubhaloMass, 'SubhaloMass', '{}/Galaxy'.format(tag),
            desc = 'Subhalo mass of the sub-group', unit = '1E10 Msun')
        fl.create_dataset(mstar, 'Mstar_30', '{}/Galaxy'.format(tag),
            desc = 'Stellar mass of the galaxy measured inside 30pkc aperture', unit = '1E10 Msun')
        fl.create_dataset(sfr_inst, 'SFR_inst_30', '{}/Galaxy'.format(tag),
            desc = 'Instantaneous sfr of the galaxy measured inside 30pkc aperture', unit = 'Msun/yr')
        fl.create_dataset(cop.T, 'COP', '{}/Galaxy'.format(tag),
            desc = 'Centre Of Potential of the galaxy', unit = 'cMpc')
        fl.create_dataset(bh_mass, 'BH_Mass', '{}/Galaxy'.format(tag),
            desc = 'Most massive black hole mass', unit = '1E10 Msun')


        del grpno, sgrpno, dnum, snum, gnum, SubhaloMass, mstar, sfr_inst, cop
        gc.collect()

        print ('Completed initial writing')

    MPI.Finalize()


if __name__ == "__main__":

    ii, tag, inp, data_folder = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


    num = str(ii)
    tag = str(tag)
    inp = str(inp)
    data_folder = str(data_folder)

    if len(num) == 1:
        num =  '0'+num


    save_to_hdf5(num, tag, inp=inp, data_folder=data_folder)
