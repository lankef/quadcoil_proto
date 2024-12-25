from scipy.io import netcdf_file
from mpl_toolkits.mplot3d import Axes3D
# Import packages.
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
from scipy import interpolate
# Importing simsopt 
from simsopt.field import CurrentPotentialSolve, CurrentPotentialFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field import Current, Coil
from simsopt.geo import CurveXYZFourier
from simsopt.objectives import SquaredFlux

def squared_flux(coils, plasma_surface):
    bs = BiotSavart(coils)
    # Evaluate on surface
    bs.set_points(plasma_surface.gamma().reshape((-1, 3)))
    return(SquaredFlux(plasma_surface, bs).J())

def coil_zeta_theta_from_cp(
    cp:CurrentPotentialFourier,
    coilsPerHalfPeriod=1,
    thetaShift=0): 

    nzeta_coil = len(cp.winding_surface.quadpoints_phi)
    # f.variables['nfp'][()]
    nfp = cp.winding_surface.nfp 
    # f.variables['theta_coil'][()]
    theta = cp.winding_surface.quadpoints_theta * 2 * np.pi 
    # f.variables['zeta_coil'][()]
    zeta = cp.winding_surface.quadpoints_phi[:nzeta_coil // nfp] * 2 * np.pi 
    # f.variables['net_poloidal_current_amperes'][()]
    net_poloidal_current_amperes = cp.net_poloidal_current_amperes 

    # ------------------------
    # Load current potential
    # ------------------------
    current_potential = np.copy(cp.Phi()[:nzeta_coil // nfp, :])\
        + cp.current_potential_secular[:nzeta_coil // nfp, :]

    if abs(net_poloidal_current_amperes) > np.finfo(float).eps:
        data = current_potential / net_poloidal_current_amperes * nfp
    else:
        data = current_potential / np.max(current_potential)

    # First apply 'roll' to be sure I use the same convention as numpy:
    theta = np.roll(theta,thetaShift)
    # Now just generate a new monotonic array with the correct first value:
    theta = theta[0] + np.linspace(0,2*np.pi,len(theta),endpoint=False)
    data = np.roll(data,thetaShift,axis=1)

    d = 2*np.pi/nfp
    zeta_3 = np.concatenate((zeta-d, zeta, zeta+d))
    data_3 = np.concatenate((data-1,data,data+1))

    # Repeat with just the contours we care about:
    contours = np.linspace(0,1,coilsPerHalfPeriod*2,endpoint=False)
    d = contours[1]-contours[0]
    contours = contours + d/2
    cdata = plt.contour(zeta_3,theta,np.transpose(data_3),contours,colors='k')
    plt.close()

    numCoilsFound = len(cdata.collections)
    if numCoilsFound != 2*coilsPerHalfPeriod:
        print("WARNING!!! The expected number of coils was not the number found.")

    contour_zeta=[]
    contour_theta=[]
    numCoils = 0
    for j in range(numCoilsFound):
        p = cdata.collections[j].get_paths()[0]
        v = p.vertices
        # Make sure the contours have increasing theta:
        if v[1,1]<v[0,1]:
            v = np.flipud(v)

        for jfp in range(nfp):
            d = 2*np.pi/nfp*jfp
            contour_zeta.append(v[:,0]+d)
            contour_theta.append(v[:,1])
            numCoils += 1
    print('numCoils', numCoils)
    print('contour_zeta', len(contour_zeta))
    print('contour_theta', len(contour_theta))
    return(contour_zeta, contour_theta)

def ifft_simsopt_legacy(x, order):
    assert len(x) >= 2*order  # the order of the fft is limited by the number of samples
    xf = rfft(x) / len(x)

    fft_0 = [xf[0].real]  # find the 0 order coefficient
    fft_cos = 2 * xf[1:order + 1].real  # find the cosine coefficients
    fft_sin = -2 * xf[:order + 1].imag  # find the sine coefficients

    combined_fft = np.concatenate([fft_sin, fft_0, fft_cos])
    return(combined_fft)

# IFFT a array in real space to a sin/cos series used by sinsopt.geo.curve
def ifft_simsopt(x, order):
    assert len(x) >= 2*order  # the order of the fft is limited by the number of samples
    xf = rfft(x) / len(x)

    fft_0 = [xf[0].real]  # find the 0 order coefficient
    fft_cos = 2 * xf[1:order + 1].real  # find the cosine coefficients
    fft_sin = (-2 * xf[:order + 1].imag)[1:]  # find the sine coefficients
    dof = np.zeros(order*2+1)
    dof[0] = fft_0[0]
    dof[1::2] = fft_sin
    dof[2::2] = fft_cos

    return(dof)

# This script assumes the contours do not zig-zag back and forth across the theta=0 line,
# after shifting the current potential by thetaShift grid points.
# The nescin file is used to provide the coil winding surface, so make sure this is consistent with the regcoil run.
# ilambda is the index in the lambda scan which you want to select.
# def cut_coil(cp, cpst):
# filename = 'regcoil_out.li383.nc' # sys.argv[1]
# TODO: these tow have a lot of duplicate code. Clean up when finalizing how QUADCOIL is integrated.
def coil_xyz_from_cp(
    cp:CurrentPotentialFourier,
    coilsPerHalfPeriod=1,
    thetaShift=0,
    save=False, save_name='placeholder'): 
    contour_zeta, contour_theta = coil_zeta_theta_from_cp(
        cp=cp,
        coilsPerHalfPeriod=coilsPerHalfPeriod,
        thetaShift=thetaShift
    )
    numCoils = len(contour_zeta)    
    nfp = cp.winding_surface.nfp 
    net_poloidal_current_amperes = cp.net_poloidal_current_amperes 

    # ------------------------
    # Load surface shape
    # ------------------------
    contour_R = []
    contour_Z = []
    for j in range(numCoils):
        contour_R.append(np.zeros_like(contour_zeta[j]))
        contour_Z.append(np.zeros_like(contour_zeta[j]))

    surf = cp.winding_surface
    for m in range(surf.mpol+1): # 0 to mpol
        for i in range(2*surf.ntor+1):
            n = i-surf.ntor
            crc = surf.rc[m, i]
            czs = surf.zs[m, i]
            if surf.stellsym:
                crs = 0
                czc = 0
            else: # Returns ValueError for stellsym cases
                crs = surf.get_rs(m, n)
                czc = surf.get_zc(m, n)
            for j in range(numCoils):
                angle = m*contour_theta[j] - n*contour_zeta[j]*surf.nfp
                # Was filled with zeroes.
                # Are lists because contou lengths are not uniform.
                contour_R[j] = contour_R[j] + crc*np.cos(angle) + crs*np.sin(angle)
                contour_Z[j] = contour_Z[j] + czs*np.sin(angle) + czc*np.cos(angle)

    contour_X = []
    contour_Y = []
    maxR = 0
    for j in range(numCoils):
        maxR = np.max((maxR,np.max(contour_R[j])))
        contour_X.append(contour_R[j]*np.cos(contour_zeta[j]))
        contour_Y.append(contour_R[j]*np.sin(contour_zeta[j]))

    coilCurrent = net_poloidal_current_amperes / numCoils

    # # Find the point of minimum separation
    # minSeparation2=1.0e+20
    # #for whichCoil1 in [5*nfp]:
    # #    for whichCoil2 in [4*nfp]:
    # for whichCoil1 in range(numCoils):
    #     for whichCoil2 in range(whichCoil1):
    #         for whichPoint in range(len(contour_X[whichCoil1])):
    #             dx = contour_X[whichCoil1][whichPoint] - contour_X[whichCoil2]
    #             dy = contour_Y[whichCoil1][whichPoint] - contour_Y[whichCoil2]
    #             dz = contour_Z[whichCoil1][whichPoint] - contour_Z[whichCoil2]
    #             separation2 = dx*dx+dy*dy+dz*dz
    #             this_minSeparation2 = np.min(separation2)
    #             if this_minSeparation2<minSeparation2:
    #                 minSeparation2 = this_minSeparation2


    if save: 
        coilsFilename='coils.'+save_name
        print("coilsFilename:",coilsFilename)
        # Write coils file
        f = open(coilsFilename,'w')
        f.write('periods '+str(nfp)+'\n')
        f.write('begin filament\n')
        f.write('mirror NIL\n')

        for j in range(numCoils):
            N = len(contour_X[j])
            for k in range(N):
                f.write('{:14.22e} {:14.22e} {:14.22e} {:14.22e}\n'.format(contour_X[j][k],contour_Y[j][k],contour_Z[j][k],coilCurrent))
            # Close the loop
            k=0
            f.write('{:14.22e} {:14.22e} {:14.22e} {:14.22e} 1 Modular\n'.format(contour_X[j][k],contour_Y[j][k],contour_Z[j][k],0))

        f.write('end\n')
        f.close()
    return(
        contour_X,
        contour_Y,
        contour_Z,
        coilCurrent,
        # np.sqrt(minSeparation2)
    )

# Load coils from lists of arrays containing x, y, and z.
def load_curves_from_xyz_legacy(
    contour_X,
    contour_Y,
    contour_Z, 
    order=None, ppp=20):

    if not order:
        order=float('inf')
        for i in range(len(contour_X)):
            xArr = contour_X[i]
            yArr = contour_Y[i]
            zArr = contour_Z[i]
            for x in [xArr, yArr, zArr]:
                if len(x)//2<order:
                    order = len(x)//2
    coil_data = []
    # Compute the Fourier coefficients for each coil
    for i in range(len(contour_X)):
        xArr = contour_X[i]
        yArr = contour_Y[i]
        zArr = contour_Z[i]

        curves_Fourier = []
        # Compute the Fourier coefficients
        for x in [xArr, yArr, zArr]:
            combined_fft = ifft_simsopt_legacy(x, order)
            curves_Fourier.append(combined_fft)

        coil_data.append(np.concatenate(curves_Fourier))

    coil_data = np.asarray(coil_data)
    coil_data = coil_data.reshape(6 * len(contour_X), order + 1)  # There are 6 * order coefficients per coil
    coil_data = np.transpose(coil_data)

    assert coil_data.shape[1] % 6 == 0
    assert order <= coil_data.shape[0]-1

    num_coils = coil_data.shape[1] // 6
    coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
    for ic in range(num_coils):
        dofs = coils[ic].dofs_matrix
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, min(order, coil_data.shape[0] - 1)):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].local_x = np.concatenate(dofs)
    return(coils)

# Load curves from lists of arrays containing x, y, and z.
def load_curves_from_xyz(
    contour_X,
    contour_Y,
    contour_Z, 
    order=None, ppp=20):
    num_coils = len(contour_X)
    # Calculating order
    if not order:
        order=float('inf')
        for i in range(num_coils):
            xArr = contour_X[i]
            yArr = contour_Y[i]
            zArr = contour_Z[i]
            for x in [xArr, yArr, zArr]:
                if len(x)//2<order:
                    order = len(x)//2
    
    coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
    # Compute the Fourier coefficients for each coil
    for ic in range(num_coils):
        xArr = contour_X[ic]
        yArr = contour_Y[ic]
        zArr = contour_Z[ic]

        # Compute the Fourier coefficients
        dofs=[]
        for x in [xArr, yArr, zArr]:
            dof_i = ifft_simsopt(x, order)
            dofs.append(dof_i)

        coils[ic].local_x = np.concatenate(dofs)
    return(coils)

# Load curves and currents 
def load_coils_from_xyz(
    contour_X,
    contour_Y,
    contour_Z, 
    coilCurrent,
    order=None, ppp=20):
    curves = load_curves_from_xyz(
        contour_X=contour_X,
        contour_Y=contour_Y,
        contour_Z=contour_Z, 
        order=order, 
        ppp=ppp
    )
    coils = []
    for curve in curves:
        coils.append(Coil(curve, Current(coilCurrent)))
    return(coils)

def cut_coil_from_cp(
    cp, coilsPerHalfPeriod, thetaShift,
    method=coil_xyz_from_cp,
    order=10, ppp=40):
    (
        contour_X,
        contour_Y,
        contour_Z,
        coilCurrent,
        # min_separation
    ) = method(
        cp=cp,
        coilsPerHalfPeriod=coilsPerHalfPeriod,
        thetaShift=thetaShift,
        save=False
    )
    coils = load_coils_from_xyz(
        contour_X,
        contour_Y,
        contour_Z, 
        coilCurrent,
        order=order,
        ppp=ppp)
    return(coils)