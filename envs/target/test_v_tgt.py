from v_tgt_field import VTgtField
import sys
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------------

#...

# --------------------------------------------------------------------
#dt = .001
dt = .5
pose_agent = np.array([0, 0, 0]) # [x, y]

vtgt_v1 = VTgtField(version=2, pose_agent=pose_agent, dt=dt)
vtgt_obj = vtgt_v1.vtgt_obj


fig,axes = plt.subplots(2,1, figsize=(8, 12))
X = vtgt_obj.map[0]
Y = vtgt_obj.map[1]
U = vtgt_obj.vtgt[0]
V = vtgt_obj.vtgt[1]
R = np.sqrt(U**2 + V**2)
q0 = axes[0].quiver(X, Y, U, V, R)
axes[0].axis('equal')

#for x, y in zip(np.linspace(0, p_sink[0], 30), np.linspace(0, p_sink[1], 30)):
#    th = 0
#for th in np.linspace(0, 90*np.pi/180, 30):
#    x = 0; y = 0
p_sink = vtgt_v1.p_sink
t_sim = 10;
x = 0; y = 0; th = 0
pose_t = np.array([[x], [y], [th]])
t0 = axes[0].text(x, y, np.array2string(pose_t, precision=3)[1:-1], fontsize=12, horizontalalignment='center', verticalalignment='center')

for t in np.arange(0, 30, dt):
    pose = np.array([x, y, th]) # [x, y, theta]
    vtgt_field_local, flag_new_target = vtgt_v1.update(pose)

    if flag_new_target:
        q0.remove()
        X = vtgt_obj.map[0]
        Y = vtgt_obj.map[1]
        U = vtgt_obj.vtgt[0]
        V = vtgt_obj.vtgt[1]
        R = np.sqrt(U**2 + V**2)
        q0 = axes[0].quiver(X, Y, U, V, R)
        axes[0].axis('equal')

    axes[0].plot(x, y, 'k.')
    t0.set_position((pose[0], pose[1]))
    pose_t = np.array([[x], [y], [th]])
    t0.set_text(np.array2string(pose_t, precision=3)[1:-1])
    
    X, Y = vtgt_obj._generate_grid(vtgt_obj.rng_get, vtgt_obj.res_get)
    U = vtgt_field_local[0]
    V = vtgt_field_local[1]
    R = np.sqrt(U**2 + V**2)
    axes[1].clear()
    axes[1].quiver(X, Y, U, V, R)
    axes[1].axis('equal')

    vtgt = vtgt_v1.get_vtgt(pose[0:2])
    axes[1].text(0, 0, np.array2string(vtgt, precision=3)[1:-1], fontsize=12, horizontalalignment='center', verticalalignment='center')

    plt.pause(0.0001)

    x += np.asscalar(vtgt[0])*dt
    y += np.asscalar(vtgt[1])*dt

    print('time: {} sec'.format(t))

plt.show()

# --------------------------------------------------------------------