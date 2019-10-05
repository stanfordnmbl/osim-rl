#from v_tgt_field import VTgtField
from envs.target import VTgtField
import sys
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------------

#...

# --------------------------------------------------------------------
version = 3 # 2: Round 1; 3: Round 2

dt = .01
#dt = .5
pose_agent = np.array([0, 0, 0]) # [x, y]

vtgt_v1 = VTgtField(version=version, dt=dt)
vtgt_v1.reset(version=version, seed=0)

t_sim = 25;
x = 0; y = 0; th = 0
pose_t = np.array([[x], [y], [th]])

for t in np.arange(0, t_sim, dt):
    pose = np.array([x, y, th]) # [x, y, theta]
    vtgt_field_local, flag_new_target = vtgt_v1.update(pose)

    vtgt = vtgt_v1.get_vtgt(pose[0:2])

    x += np.asscalar(vtgt[0])*dt
    y += np.asscalar(vtgt[1])*dt

    if flag_new_target:
        print('time: {} sec'.format(t))
        print('New Target!!!')

