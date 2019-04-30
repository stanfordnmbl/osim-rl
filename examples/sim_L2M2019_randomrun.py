from osim.env import L2M2019Env
import numpy as np

env = L2M2019Env(seed=5, difficulty=2)
env.change_model(model='3D', difficulty=2, seed=11)
observation = env.reset(project=False, seed=11)

# visualize v_tgt --------------------------------------------------------------
import matplotlib.pyplot as plt
flag_visualize_vtgt = 1
if flag_visualize_vtgt:
    vtgt_obj = env.vtgt.vtgt_obj
    fig,axes = plt.subplots(2,1, figsize=(8, 12))
    X = vtgt_obj.map[0]
    Y = vtgt_obj.map[1]
    U = vtgt_obj.vtgt[0]
    V = vtgt_obj.vtgt[1]
    R = np.sqrt(U**2 + V**2)
    q0 = axes[0].quiver(X, Y, U, V, R)
    axes[0].axis('equal')
    pose_t = np.array([[0], [0], [0]])
    t0 = axes[0].text(0, 0, np.array2string(pose_t, precision=3)[1:-1], fontsize=12, horizontalalignment='center', verticalalignment='center')
# visualize v_tgt --------------------------------------------------------------

for i in range(300):
    observation, reward, done, info=env.step(env.action_space.sample(), project=True)
    obs_dict = env.get_observation_dict()
    if done:
        break

# visualize v_tgt --------------------------------------------------------------
    if flag_visualize_vtgt:
        if env.flag_new_v_tgt_field:
            q0.remove()
            X = vtgt_obj.map[0]
            Y = vtgt_obj.map[1]
            U = vtgt_obj.vtgt[0]
            V = vtgt_obj.vtgt[1]
            R = np.sqrt(U**2 + V**2)
            q0 = axes[0].quiver(X, Y, U, V, R)
            axes[0].axis('equal')

        pose = env.pose
            
        axes[0].plot(pose[0], pose[1], 'k.')
        t0.set_position((pose[0], pose[1]))
        pose_t = np.array([[pose[0]], [pose[1]], [pose[2]*180/np.pi]])
        t0.set_text(np.array2string(pose_t, precision=3)[1:-1])
        
        X, Y = vtgt_obj._generate_grid(vtgt_obj.rng_get, vtgt_obj.res_get)
        U = env.v_tgt_field[0]
        V = env.v_tgt_field[1]
        R = np.sqrt(U**2 + V**2)
        axes[1].clear()
        axes[1].quiver(X, Y, U, V, R)
        axes[1].axis('equal')

        vtgt = env.vtgt.get_vtgt(pose[0:2])
        axes[1].text(0, 0, np.array2string(vtgt, precision=3)[1:-1], fontsize=12, horizontalalignment='center', verticalalignment='center')

        plt.pause(0.0001)
# visualize v_tgt --------------------------------------------------------------
if flag_visualize_vtgt:
    plt.show()
