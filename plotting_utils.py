# Import packages.
import os
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import interpolate
from matplotlib import cm, colors
# Importing simsopt 
from simsopt.field import CurrentPotentialFourier

def plot_coil_Phi_IG(
    cp_opt:CurrentPotentialFourier, 
    nlevels=40, 
    plot_sv_only=False, 
    plot_2d_contour=False):
    '''
    Plots a coil configuration on 3d surface. 
    Parameters:
    
    `nlevels` - # contour levels
    `plot_sv_only` - When True, plots Phi_sv only 
    `plot_2d_contour` - When True, plots 2d contour also 
    '''
    theta1d, phi1d = cp_opt.quadpoints_theta, cp_opt.quadpoints_phi
    theta2d, phi2d = np.meshgrid(theta1d, phi1d)

    # Creating interpolation for mapping 2d contours onto 3d surface
    gamma=cp_opt.winding_surface.gamma()
    # Wrapping gamma and theta for periodic interpolation
    gamma_periodic = np.pad(gamma, ((0,1), (0,1), (0,0)), 'wrap')
    theta1d_periodic = np.concatenate((theta1d, [1+theta1d[0]]))
    phi1d_periodic = np.concatenate((phi1d, [1+phi1d[0]]))
    # Wrapped meshgrid
    theta2d_periodic, phi2d_periodic = np.meshgrid(theta1d_periodic, phi1d_periodic)
    # Creating interpolation
    phi_theta_to_xyz = interpolate.LinearNDInterpolator(
        np.array([phi2d_periodic.flatten(), theta2d_periodic.flatten()]).T,
        gamma_periodic.reshape(-1, 3)
    )

    # Calculating Phi
    G = cp_opt.net_poloidal_current_amperes
    I = cp_opt.net_toroidal_current_amperes  
    if plot_sv_only:
        Phi = cp_opt.Phi()  
    else:    
        Phi = cp_opt.Phi() \
            + phi2d*G \
            + theta2d*I


    # Making 2d contour plot   
    quad_contour_set = plt.contour(
        phi2d,
        theta2d, 
        Phi,
        levels=nlevels,
        algorithm='threaded'
    )
    if plot_2d_contour:
        plt.show()
    else:
        plt.close()
    fig = plt.figure()
    fig.set_dpi(400)
    ax = fig.add_subplot(projection='3d')

    norm = colors.Normalize(vmin=np.min(Phi), vmax=np.max(Phi), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
    level_color = mapper.to_rgba(quad_contour_set.levels)

    # Phi contours:

    for i in range(len(quad_contour_set.allsegs)):
        # Loop over all contour levels
        seg_i = quad_contour_set.allsegs[i]
        kind_i = quad_contour_set.allkinds[i]
        # Splitting each contour level by vertex kind.
        # only kind=2 should be plotted.
        seg_i[kind_i==1] = np.nan
        list_of_segments = [
            list(g) for m, g in itertools.groupby(
                seg_i, key=lambda x: not np.all(np.isnan(x))
            ) if m
        ]
        for segment in list_of_segments:
            # If there are more than 0 segments for the level
            # Mapping the segments' phi, theta coordinate into xyz
            xyz_seg_i = phi_theta_to_xyz(segment)
            ax.plot(
                xyz_seg_i[:,0], 
                xyz_seg_i[:,1], 
                xyz_seg_i[:,2],
                # facecolors=facecolors
                c=level_color[i],
                linewidth = 0.5
            )

    ax.axis('equal')
    return(fig, ax)

def plot_trade_off(
        Phi_list, 
        f_x, f_y, 
        xlabel, ylabel, 
        **kwargs
    ):
    x_list = list(map(f_x, Phi_list))
    y_list = list(map(f_y, Phi_list))
    plt.scatter(x_list, y_list, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def generate_video(cp, Phi_list, vid_name):
    # img_list = []
    for i in range(len(Phi_list)):
        cp_opt_temp_cp = CurrentPotentialFourier(
            cp.winding_surface, 
            cp.net_poloidal_current_amperes,
            cp.net_toroidal_current_amperes, 
            cp.nfp, 
            cp.stellsym,
            cp.mpol, 
            cp.ntor,
        )
        cp_opt_temp_cp.set_dofs(Phi_list[i]) # i_best_both
        fig, ax = plot_coil_Phi_IG(
            cp_opt=cp_opt_temp_cp, 
            nlevels=100, 
            plot_sv_only=False, 
            plot_2d_contour=False
        )
        # img_list.append(ax)
    # ani = animation.ArtistAnimation(fig, img_list, interval=50, blit=True,
    #                         repeat_delay=1000)
        plt.savefig("%05d.vid_temp_file.png" % i, dpi=500)

    subprocess.call([
        'ffmpeg', '-y', '-framerate', '20', '-i', '%05d.vid_temp_file.png', '-r', '15', '-pix_fmt', 'yuv420p',
        vid_name+'.mp4'
    ])
    for file_name in glob.glob("*.vid_temp_file.png"):
        os.remove(file_name)
