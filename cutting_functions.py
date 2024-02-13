# Import packages.
import numpy as np
# The staircase always average to y=x so 
# that phi is not scaled.
# x_width must be within (0, x_step).
# Setting x_width of 0 will make the grad zero,
# while it should've been a series of delta funcs.
def staircase(
    x:np.float64,
    # The "period" of the staircase
    x_step=2,
    # The width of slopes
    x_width=0.5,
    # The location of the center of the first plateau
    x_phase=1):
    x_base = x_step*np.floor((x-x_phase)/x_step)
    x_residual = x-x_base-x_phase
    stair_residual = np.interp(
        x=x_residual,
        xp=np.array([0, x_step/2-x_width/2, x_step/2+x_width/2, x_step]),
        fp=np.array([0, 0, x_step, x_step])
    )
    return(x_base+stair_residual+x_phase)