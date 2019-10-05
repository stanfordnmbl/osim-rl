#from v_tgt_field import VTgtField
from envs.target import VTgtField
import sys
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------------

#...

# --------------------------------------------------------------------
#dt = .001
dt = .5
pose_agent = np.array([0, 0, 0]) # [x, y]

vtgt_v1 = VTgtField(version=2, dt=dt)
vtgt_v1.reset(version=2, seed=0)

p_sink = vtgt_v1.p_sink
t_sim = 10;
x = 0; y = 0; th = 0
pose_t = np.array([[x], [y], [th]])

for t in np.arange(0, 30, dt):
    pose = np.array([x, y, th]) # [x, y, theta]
    vtgt_field_local, flag_new_target = vtgt_v1.update(pose)

    vtgt = vtgt_v1.get_vtgt(pose[0:2])

    x += np.asscalar(vtgt[0])*dt
    y += np.asscalar(vtgt[1])*dt

    print('time: {} sec'.format(t))

#plt.show()

# --------------------------------------------------------------------