# Author(s): Seungmoon Song <seungmoon.song@gmail.com>
"""
adapted from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np
from envs.control.loco_reflex_song2019 import LocoCtrl

class OsimReflexCtrl(object):
    # from gait14dof22musc_planar_20170320
    Fmax_RF = 2191.74098360656
    Fmax_VAS = 9593.95082
    Fmax_GAS = 4690.57377
    Fmax_SOL = 7924.996721

    Fmax_ABD = 4460.290481
    Fmax_ADD = 3931.8

    def __init__(self, dt=0.01, mode='3D', prosthetic=False):
        self.dt = dt
        self.t = 0
        self.mode = mode
        self.prosthetic = prosthetic
        self.mass = 75 #approximate for intact model
        self.g = 9.81 # gravity

        if self.mode is '3D':
            self.n_par = len(LocoCtrl.cp_keys)
            control_dimension = 3
        elif self.mode is '2D':
            self.n_par = 37
            control_dimension = 2
        self.cp_map = LocoCtrl.cp_map
        self.ctrl = LocoCtrl(self.dt, control_dimension=control_dimension, params=np.ones(self.n_par), prosthetic=self.prosthetic)
        self.par_space = self.ctrl.par_space

# -----------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.ctrl.reset()

# -----------------------------------------------------------------------------------------------------------------
    def update(self, obs):
        self.t += self.dt
        self.ctrl.update(self._obs2reflexobs(obs))
        #for s_l in ['r', 'l']:
        #    str_control_phase = s_l + ':'
        #    for ctrl_ph in self.ctrl.spinal_control_phase['{}_leg'.format(s_l)]:
        #        if self.ctrl.spinal_control_phase['{}_leg'.format(s_l)][ctrl_ph]:
        #            str_control_phase = str_control_phase + '  ' + ctrl_ph
        #    print(str_control_phase)
        return self._reflexstim2stim()

# -----------------------------------------------------------------------------------------------------------------
    def set_control_params(self, params):
        self.ctrl.set_control_params(params)

# -----------------------------------------------------------------------------------------------------------------
    def set_control_params_RL(self, s_leg, params):
        self.ctrl.set_control_params_RL(s_leg, params)

# -----------------------------------------------------------------------------------------------------------------
    def _obs2reflexobs(self, obs):
        # refer to LocoCtrl.s_b_keys and LocoCtrl.s_l_keys
        # osim coordinate
        #   [0] x: forward
        #   [1] y: upward
        #   [2] z: rightward

        sensor_data = {'body':{}, 'r_leg':{}, 'l_leg':{}}
        # !!! todo:
        # !!! need to check angle in frontal plane
        sensor_data['body']['theta'] = [obs['joint_pos']['ground_pelvis'][1],
                                        -obs['joint_pos']['ground_pelvis'][0],
                                        obs['joint_pos']['ground_pelvis'][2] ]
            # theta[0]: around local x axis (pointing anterior)
            # theta[1]: around local y axis (pointing leftward)
            # theta[2]: around local z axis (pointing upward)
        sensor_data['body']['d_pos'] = [obs['joint_vel']['ground_pelvis'][3],
                                        obs['joint_vel']['ground_pelvis'][4],
                                        obs['joint_vel']['ground_pelvis'][5] ]
            # pos[0]: local x
            # pos[1]: local y
            # pos[2]: local z
        sensor_data['body']['dtheta'] = [obs['joint_vel']['ground_pelvis'][1],
                                        -obs['joint_vel']['ground_pelvis'][0],
                                        obs['joint_vel']['ground_pelvis'][2] ]

        #!!! check grf...
        #!!! len(obs['forces']['pros_foot_r_0']) == 18
        #!!! len(obs['forces']['pros_foot_r_0']) == 24
        #temp_list = list(obs['forces']['pros_foot_r_0'][i] for i in [1, 7, 13])
        #temp_list = list(obs['forces']['foot_r'][i] for i in [1, 7, 13, 19])
        #print(temp_list)
        #temp_list = list(obs['forces']['foot_l'][i] for i in [1, 7, 13, 19])
        #print(temp_list)
        # !!!hack!!! should calculate GRF correctly
        #!!! scale grf...
        if self.prosthetic is True:
            sensor_data['r_leg']['load_ipsi'] = -obs['forces']['pros_foot_r_0'][1]/(self.mass*self.g)
        else:
            sensor_data['r_leg']['load_ipsi'] = -obs['forces']['foot_r'][1]/(self.mass*self.g)
        sensor_data['l_leg']['load_ipsi'] = -obs['forces']['foot_l'][1]/(self.mass*self.g)
        for s_leg in ['r_leg', 'l_leg']:
            s_legc = 'l_leg' if s_leg is 'r_leg' else 'r_leg'
            s_l = 'r' if s_leg is 'r_leg' else 'l'

            sensor_data[s_leg]['contact_ipsi'] = 1 if sensor_data[s_leg]['load_ipsi'] > 0.1 else 0
            sensor_data[s_leg]['contact_contra'] = 1 if sensor_data[s_legc]['load_ipsi'] > 0.1 else 0
            sensor_data[s_leg]['load_contra'] = sensor_data[s_legc]['load_ipsi']

            sensor_data[s_leg]['phi_hip'] = -obs['joint_pos']['hip_{}'.format(s_l)][0] + np.pi
            sensor_data[s_leg]['phi_knee'] = obs['joint_pos']['knee_{}'.format(s_l)][0] + np.pi
            sensor_data[s_leg]['phi_ankle'] = -obs['joint_pos']['ankle_{}'.format(s_l)][0] + .5*np.pi
            sensor_data[s_leg]['dphi_knee'] = obs['joint_vel']['knee_{}'.format(s_l)][0]

            # alpha = hip - 0.5*knee
            sensor_data[s_leg]['alpha'] = sensor_data[s_leg]['phi_hip'] - .5*sensor_data[s_leg]['phi_knee']
            dphi_hip = obs['joint_vel']['hip_{}'.format(s_l)][2]
            sensor_data[s_leg]['dalpha'] = dphi_hip - .5*sensor_data[s_leg]['dphi_knee']
            sensor_data[s_leg]['alpha_f'] = obs['joint_pos']['hip_{}'.format(s_l)][1] + .5*np.pi
            # !!! need to check alpha_f (left and right)

            sensor_data[s_leg]['F_RF'] = obs['muscles']['rect_fem_{}'.format(s_l)]['fiber_force']/self.Fmax_RF
            sensor_data[s_leg]['F_VAS'] = obs['muscles']['vasti_{}'.format(s_l)]['fiber_force']/self.Fmax_VAS

            if self.prosthetic is False or s_leg is 'l_leg':
                sensor_data[s_leg]['F_GAS'] = obs['muscles']['gastroc_{}'.format(s_l)]['fiber_force']/self.Fmax_GAS
                sensor_data[s_leg]['F_SOL'] = obs['muscles']['soleus_{}'.format(s_l)]['fiber_force']/self.Fmax_SOL

        return sensor_data

# -----------------------------------------------------------------------------------------------------------------
    def _reflexstim2stim(self):
        stim = [self.ctrl.stim['r_leg']['HAM'], # (hamstring_r)
                self.ctrl.stim['r_leg']['BFSH'], # (bifemsh_r)
                self.ctrl.stim['r_leg']['GLU'], # (glut_max_r)
                self.ctrl.stim['r_leg']['HFL'], # (iliopsoas_r)
                self.ctrl.stim['r_leg']['RF'], # (rect_fem_r)
                self.ctrl.stim['r_leg']['VAS'], # (vasti_r)
                self.ctrl.stim['l_leg']['HAM'], # (hamstring_l)
                self.ctrl.stim['l_leg']['BFSH'], # (bifemsh_l)
                self.ctrl.stim['l_leg']['GLU'], # (glut_max_l)
                self.ctrl.stim['l_leg']['HFL'], # (iliopsoas_l)
                self.ctrl.stim['l_leg']['RF'], # (rect_fem_l)
                self.ctrl.stim['l_leg']['VAS'], # (vasti_l)
                self.ctrl.stim['l_leg']['GAS'], # (gastroc_l)
                self.ctrl.stim['l_leg']['SOL'], # (soleus_l)
                self.ctrl.stim['l_leg']['TA'] ] # (tib_ant_l)

        if self.mode is '3D':
            stim.insert(0, self.ctrl.stim['r_leg']['HAB']) # (abd_r)
            stim.insert(1, self.ctrl.stim['r_leg']['HAD']) # (add_r)
            stim.insert(8, self.ctrl.stim['l_leg']['HAB']) # (abd_r)
            stim.insert(9, self.ctrl.stim['l_leg']['HAD']) # (add_r)
        elif self.mode is '2D':
            stim.insert(0, .1)
            stim.insert(1, .1*self.Fmax_ADD/self.Fmax_ABD)
            stim.insert(8, .1)
            stim.insert(9, .1*self.Fmax_ADD/self.Fmax_ABD)


        if self.prosthetic is False:
            stim.insert(8, self.ctrl.stim['r_leg']['GAS'])
            stim.insert(9, self.ctrl.stim['r_leg']['SOL'])
            stim.insert(10, self.ctrl.stim['r_leg']['TA'])
        return stim
