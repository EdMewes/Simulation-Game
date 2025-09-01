import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py as h5
from physics_sim.keys import numeric
import os

def dedalus(file_name, only_first:bool=True):
    with h5.File(file_name, 'r') as f:

        temp = f[f'tasks/temp'][:] 
    rows = temp.shape[0]
    file_num = numeric(file_name)
    if only_first == True:
        plt.figure()
        plt.imshow(temp[0], origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Temp')
        plt.title(f'Temperature field')
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


def frame_lookup(global_frame, files, frame_count):
    for index, snapshot_count in enumerate(frame_count):
        if global_frame < snapshot_count:
            return files[index], global_frame
        global_frame -= snapshot_count
    raise IndexError


def animate(files, frame_count, snapshot_rate ,save_file_path):
    # Create first frame
    fig, ax = plt.subplots()
    with h5.File(files[0], "r") as f:
        first = f["tasks"]["temp"][0]
        im = ax.imshow(first, origin="lower", cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax)

    def update(global_frame):
        file, index = frame_lookup(global_frame, files, frame_count)
        with h5.File(file, "r") as f:
            data = f["tasks"]["temp"][index]
            im.set_array(data)
            # im.set_clim(vmin=data.min(), vmax=data.max())
            ax.set_title(f"{os.path.basename(file)} frame {index}")
            return [im]
        
    
    total_frames = sum(frame_count)    
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/snapshot_rate, blit=True)
    ani.save(save_file_path)



    return 0