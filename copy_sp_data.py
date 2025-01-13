import h5py
import numpy

tags = ['011_z004p770',  '010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000', '004_z011p000', '003_z012p000', '002_z013p000', '001_z014p000', '000_z015p000']

for rg in range(40):

    rg = str(rg)
    if len(str(rg))==1:
        rg = "0" + rg
    print(f"Region {rg}")

    new_f = f"data/FLARES_{rg}_sp_info.hdf5"
    old_f = f"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data1/FLARES_{rg}_sp_info.hdf5"

    with h5py.File(new_f, 'a') as nf, h5py.File(old_f, 'r') as of:
        for tag in tags:
            # path = f"{tag}/Galaxy/Central_Indices"
            of.copy(of[f"{tag}/Galaxy/Indices"], nf, f"{tag}/Galaxy/Indices")
            of.copy(of[f"{tag}/Galaxy/Central_Indices"], nf, f"{tag}/Galaxy/Central_Indices")
            #of.copy(of[f"{tag}/Particle/DM_Index"], nf, f"{tag}/Particle/DM_Index")
            #of.copy(of[f"{tag}/Particle/S_Index"], nf, f"{tag}/Particle/S_Index")
            #of.copy(of[f"{tag}/Particle/G_Index"], nf, f"{tag}/Particle/G_Index")
            #of.copy(of[f"{tag}/Particle/BH_Index"], nf, f"{tag}/Particle/BH_Index")


print("Done :)")
