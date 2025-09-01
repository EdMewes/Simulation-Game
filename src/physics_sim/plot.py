import matplotlib.pyplot as plt
import h5py as h5
from physics_sim.keys import numeric

def dedalus(file_name, only_first:bool=True):
    with h5.File(file_name, 'r') as f:

        temp = f[f'tasks/temp'][:] 
    rows = temp.shape[0]
    file_num = numeric(file_name)
    if only_first == True:
        plt.figure()
        plt.imshow(temp[i], origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Temp')
        plt.title(f'Temperature field: {i}')
        plt.xlabel('X grid')
        plt.ylabel('Y grid')
        plt.savefig(f'fig/snapshot_s{(file_num)}.png', dpi = 10)
        plt.close()
    else:
        for i in range(rows):   
            plt.figure()
            plt.imshow(temp[i], origin='lower', cmap='viridis', aspect='auto')
            plt.colorbar(label='Temp')
            plt.title(f'Temperature field: {i}')
            plt.xlabel('X grid')
            plt.ylabel('Y grid')
            plt.savefig(f'fig/snapshot_s{(file_num)}_{i}.png', dpi = 10)
            plt.close()
    return 0