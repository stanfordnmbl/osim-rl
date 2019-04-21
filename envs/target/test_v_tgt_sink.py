from v_tgt_field import VTgtSink
import sys
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------------

#...

# --------------------------------------------------------------------
rng_xy = np.array([[-20, 20], [-20, 20]])
#rng_xy = np.array([[-10, 10], [-5, 5]])
vtgt_obj = VTgtSink(rng_xy, res_map=np.array([1, 1]), res_get=np.array([3, 1]))

p_sink = np.array([13.8,2.5]) # [x, y]
d_sink = np.linalg.norm(p_sink)
v_amp_rng = np.array([1.0, 2.0])
vtgt_obj.create_vtgt_sink(p_sink, d_sink, v_amp_rng, v_phase0=np.pi)


fig,axes = plt.subplots(2,1, figsize=(8, 12))
X = vtgt_obj.map[0]
Y = vtgt_obj.map[1]
U = vtgt_obj.vtgt[0]
V = vtgt_obj.vtgt[1]
R = np.sqrt(U**2 + V**2)
axes[0].quiver(X, Y, U, V, R)
axes[0].axis('equal')


pose = np.array([0.0, 0.0, 0*np.pi/180]) # [x, y, theta]
vtgt = vtgt_obj.get_vtgt(pose[0:2])

print('vtgt: {}'.format(vtgt))

vtgt_field_local = vtgt_obj.get_vtgt_field_local(pose)

X, Y = vtgt_obj._generate_grid(vtgt_obj.rng_get, vtgt_obj.res_get)
U = vtgt_field_local[0]
V = vtgt_field_local[1]
R = np.sqrt(U**2 + V**2)
axes[1].quiver(X, Y, U, V, R)
axes[1].axis('equal')

#for x, y in zip(np.linspace(0, p_sink[0], 30), np.linspace(0, p_sink[1], 30)):
#    th = 0
#for th in np.linspace(0, 90*np.pi/180, 30):
#    x = 0; y = 0
for x, y, th in zip(np.linspace(0, p_sink[0], 30), np.linspace(0, p_sink[1], 30), np.linspace(0, 90*np.pi/180, 30)):
    pose = np.array([x, y, th]) # [x, y, theta]
    vtgt_field_local = vtgt_obj.get_vtgt_field_local(pose)

    axes[0].plot(x, y, 'k.')

    X, Y = vtgt_obj._generate_grid(vtgt_obj.rng_get, vtgt_obj.res_get)
    U = vtgt_field_local[0]
    V = vtgt_field_local[1]
    R = np.sqrt(U**2 + V**2)
    axes[1].clear()
    axes[1].quiver(X, Y, U, V, R)
    axes[1].axis('equal')

    plt.pause(0.0001)

plt.show()



# --------------------------------------------------------------------