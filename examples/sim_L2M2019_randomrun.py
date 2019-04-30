from osim.env import L2M2019Env
import numpy as np

seed=None
env = L2M2019Env(seed=seed, difficulty=2)
env.change_model(model='3D', difficulty=2, seed=seed)
observation = env.reset(project=False, seed=seed)

# visualize v_tgt --------------------------------------------------------------
import matplotlib.pyplot as plt
flag_visualize_vtgt = 1
if flag_visualize_vtgt:
    vtgt_obj = env.vtgt.vtgt_obj
    fig,axes = plt.subplots(2,1, figsize=(4, 6))
    X = vtgt_obj.map[0]
    Y = vtgt_obj.map[1]
    U = vtgt_obj.vtgt[0]
    V = vtgt_obj.vtgt[1]
    R = np.sqrt(U**2 + V**2)
    q0 = axes[0].quiver(X, Y, U, V, R)
    axes[0].axis('equal')
# visualize v_tgt --------------------------------------------------------------

for i in range(300):
    observation, reward, done, info=env.step(env.action_space.sample(), project=True)
    obs_dict = env.get_observation_dict()
    if done:
        break

# visualize v_tgt --------------------------------------------------------------
    if flag_visualize_vtgt and i%20==0:
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
        
        X, Y = vtgt_obj._generate_grid(vtgt_obj.rng_get, vtgt_obj.res_get)
        U = env.v_tgt_field[0]
        V = env.v_tgt_field[1]
        R = np.sqrt(U**2 + V**2)
        axes[1].clear()
        axes[1].quiver(X, Y, U, V, R)
        axes[1].plot(0, 0, 'k.')
        axes[1].axis('equal')
        
        plt.pause(0.0001)
# visualize v_tgt --------------------------------------------------------------
if flag_visualize_vtgt:
    plt.show()