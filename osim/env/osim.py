import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random

## OpenSim interface
# The amin purpose of this class is to provide wrap all 
# the necessery elements of OpenSim in one place
# The actual RL environment then only needs to:
# - open a model
# - actuate
# - integrate
# - read the high level description of the state
# The objective, stop condition, and other gym-related
# methods are enclosed in the OsimEnv class
class OsimModel(object):
    # Initialize simulation
    stepsize = 0.01

    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None
    verbose = False
    istep = 0
    
    state_desc_istep = None
    prev_state_desc = None
    state_desc = None
    integrator_accuracy = None

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize, integrator_accuracy = 5e-5):
        self.integrator_accuracy = integrator_accuracy
        self.model = opensim.Model(model_path)
        self.model_state = self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()

        if self.verbose:
            self.list_elements()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.noutput = self.muscleSet.getSize()
            
        self.model.addController(self.brain)
        self.model_state = self.model.initSystem()

    def list_elements(self):
        print("JOINTS")
        for i in range(self.jointSet.getSize()):
            print(i,self.jointSet.get(i).getName())
        print("\nBODIES")
        for i in range(self.bodySet.getSize()):
            print(i,self.bodySet.get(i).getName())
        print("\nMUSCLES")
        for i in range(self.muscleSet.getSize()):
            print(i,self.muscleSet.get(i).getName())
        print("\nFORCES")
        for i in range(self.forceSet.getSize()):
            print(i,self.forceSet.get(i).getName())
        print("\nMARKERS")
        for i in range(self.markerSet.getSize()):
            print(i,self.markerSet.get(i).getName())

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")

        action = np.clip(np.array(action), 0.0, 1.0)
        self.last_action = action
            
        brain = opensim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    """
    Directly modifies activations in the current state.
    """
    def set_activations(self, activations):
        if np.any(np.isnan(activations)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")
        for j in range(self.muscleSet.getSize()):
            self.muscleSet.get(j).setActivation(self.state, activations[j])
        self.reset_manager()

    """
    Get activations in the given state.
    """
    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in range(joint.numCoordinates())]

        ## Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)]
            
            res["body_pos_rot"][name] = [body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)]

        ## Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html 
        
        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        ## Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_acc"] = [self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)]

        return res

    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def get_action_space_size(self):
        return self.noutput

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self):
        self.state = self.model.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)
        self.istep = 0

        self.reset_manager()

    def get_state(self):
        return opensim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.istep = int(self.state.getTime() / self.stepsize) # TODO: remove istep altogether
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

        # Integrate till the new endtime
        self.state = self.manager.integrate(self.stepsize * self.istep)


class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 300

## OpenAI interface
# The amin purpose of this class is to provide wrap all 
# the functions of OpenAI gym. It is still an abstract
# class but closer to OpenSim. The actual classes of
# environments inherit from this one and:
# - select the model file
# - define the rewards and stopping conditions
# - define an obsernvation as a function of state
class OsimEnv(gym.Env):
    action_space = None
    observation_space = None
    osim_model = None
    istep = 0
    verbose = False

    visualize = False
    spec = None
    time_limit = 1e10

    prev_state_desc = None

    model_path = None # os.path.join(os.path.dirname(__file__), '../models/MODEL_NAME.osim')    

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : None
    }

    def get_reward(self):
        raise NotImplementedError

    def is_done(self):
        return False

    def __init__(self, visualize = True, integrator_accuracy = 5e-5):
        self.visualize = visualize
        self.integrator_accuracy = integrator_accuracy
        self.load_model()

    def load_model(self, model_path = None):
        if model_path:
            self.model_path = model_path
            
        self.osim_model = OsimModel(self.model_path, self.visualize, integrator_accuracy = self.integrator_accuracy)

        # Create specs, action and observation spaces mocks for compatibility with OpenAI gym
        self.spec = Spec()
        self.spec.timestep_limit = self.time_limit

        self.action_space = ( [0.0] * self.osim_model.get_action_space_size(), [1.0] * self.osim_model.get_action_space_size() )
#        self.observation_space = ( [-math.pi*100] * self.get_observation_space_size(), [math.pi*100] * self.get_observation_space_s
        self.observation_space = ( [0] * self.get_observation_space_size(), [0] * self.get_observation_space_size() )
        
        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)

    def get_state_desc(self):
        return self.osim_model.get_state_desc()

    def get_prev_state_desc(self):
        return self.prev_state_desc

    def get_observation(self):
        # This one will normally be overwrtitten by the environments
        # In particular, for the gym we want a vector and not a dictionary
        return self.osim_model.get_state_desc()

    def get_observation_dict(self):
        return self.osim_model.get_state_desc()

    def get_observation_space_size(self):
        return 0

    def get_action_space_size(self):
        return self.osim_model.get_action_space_size()

    def reset(self, project=True, obs_as_dict=True):
        self.osim_model.reset()

        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return get_observation_dict()
        return self.get_observation()

    def step(self, action, project=True, obs_as_dict=True):
        self.prev_state_desc = self.get_state_desc()        
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        if project:
            if obs_as_dict:
                obs = self.get_observation_dict()
            else:
                obs = self.get_observation()
        else:
            obs = self.get_state_desc()
            
        return [ obs, self.get_reward(), self.is_done() or (self.osim_model.istep >= self.spec.timestep_limit), {} ]

    def render(self, mode='human', close=False):
        return


class L2M2019Env(OsimEnv):
# to change later:
# muscle v: normalize by max_contraction_velocity, 15 lopt / s
    model = '3D'

    # from gait14dof22musc_20170320.osim
    MASS = 75.16460000000001 # 11.777 + 2*(9.3014 + 3.7075 + 0.1 + 1.25 + 0.2166) + 34.2366
    G = 9.80665 # from gait1dof22muscle

    LENGTH0 = 1 # leg length

    footstep = {}
    footstep['n'] = 0
    footstep['new'] = False
    footstep['r_contact'] = 0
    footstep['l_contact'] = 0

    dict_muscle = { 'abd': 'HAB',
                    'add': 'HAD',
                    'iliopsoas': 'HFL',
                    'glut_max': 'GLU',
                    'hamstrings': 'HAM',
                    'rect_fem': 'RF',
                    'vasti': 'VAS',
                    'bifemsh': 'BFSH',
                    'gastroc': 'GAS',
                    'soleus': 'SOL',
                    'tib_ant': 'TA'}

    act2mus = [0, 1, 4, 7, 3, 2, 5, 6, 8, 9, 10, 11, 12, 15, 18, 14, 13, 16, 17, 19, 20, 21]
    # maps muscle order in action to muscle order in gait14dof22musc_20170320.osim
    # muscle order in action
    #    HAB, HAD, HFL, GLU, HAM, RF, VAS, BFSH, GAS, SOL, TA 
    # muscle order in gait14dof22musc_20170320.osim
    #    HAB, HAD, HAM, BFSH, GLU, HFL, RF, VAS, GAS, SOL, TA
    #    or abd, add, hamstrings, bifemsh, glut_max, iliopsoas, rect_fem, vasti, gastroc, soleus, tib_ant

    INIT_POSE = np.array([
        0, # forward speed
        0, # rightward speed
        0.94, # pelvis height
        0*np.pi/180, # trunk lean
        0*np.pi/180, # [right] hip adduct
        0*np.pi/180, # hip flex
        0*np.pi/180, # knee extend
        0*np.pi/180, # ankle flex
        0*np.pi/180, # [left] hip adduct
        0*np.pi/180, # hip flex
        0*np.pi/180, # knee extend
        0*np.pi/180]) # ankle flex

    obs_vtgt_space = np.array([[-10] * 2*11*11, [10] * 2*11*11])

    obs_body_space = np.array([[-1.0] * 97, [1.0] * 97])
    obs_body_space[:,0] = [0, 3] # pelvis height
    obs_body_space[:,1] = [-np.pi, np.pi] # pelvis pitch
    obs_body_space[:,2] = [-np.pi, np.pi] # pelvis roll
    obs_body_space[:,3] = [-20, 20] # pelvis vel (forward)
    obs_body_space[:,4] = [-20, 20] # pelvis vel (leftward)
    obs_body_space[:,5] = [-20, 20] # pelvis vel (upward)
    obs_body_space[:,6] = [-10*np.pi, 10*np.pi] # pelvis angular vel (pitch)
    obs_body_space[:,7] = [-10*np.pi, 10*np.pi] # pelvis angular vel (roll)
    obs_body_space[:,8] = [-10*np.pi, 10*np.pi] # pelvis angular vel (yaw)
    obs_body_space[:,[9 + x for x in [0, 44]]] = np.array([[-5, 5]]).transpose() # (r,l) ground reaction force normalized to bodyweight (forward)
    obs_body_space[:,[10 + x for x in [0, 44]]] = np.array([[-5, 5]]).transpose() # (r, l) ground reaction force normalized to bodyweight (rightward)
    obs_body_space[:,[11 + x for x in [0, 44]]] = np.array([[-10, 10]]).transpose() # (r, l) ground reaction force normalized to bodyweight (upward)
    obs_body_space[:,[12 + x for x in [0, 44]]] = np.array([[-45*np.pi/180, 90*np.pi/180]]).transpose() # (r, l) joint: (+) hip abduction
    obs_body_space[:,[13 + x for x in [0, 44]]] = np.array([[-180*np.pi/180, 45*np.pi/180]]).transpose() # (r, l) joint: (+) hip extension
    obs_body_space[:,[14 + x for x in [0, 44]]] = np.array([[-180*np.pi/180, 0]]).transpose() # (r, l) joint: (+) knee extension
    obs_body_space[:,[15 + x for x in [0, 44]]] = np.array([[-45*np.pi/180, 90*np.pi/180]]).transpose() # (r, l) joint: (+) ankle extension (plantarflexion)
    obs_body_space[:,[16 + x for x in [0, 44]]] = np.array([[-5*np.pi, 5*np.pi]]).transpose() # (r, l) joint: (+) hip abduction
    obs_body_space[:,[17 + x for x in [0, 44]]] = np.array([[-5*np.pi, 5*np.pi]]).transpose() # (r, l) joint: (+) hip extension
    obs_body_space[:,[18 + x for x in [0, 44]]] = np.array([[-5*np.pi, 5*np.pi]]).transpose() # (r, l) joint: (+) knee extension
    obs_body_space[:,[19 + x for x in [0, 44]]] = np.array([[-5*np.pi, 5*np.pi]]).transpose() # (r, l) joint: (+) ankle extension (plantarflexion)
    obs_body_space[:,[20 + x for x in list(range(0, 33, 3)) + list(range(44, 77, 3))]] = np.array([[0, 2]]).transpose() # (r, l) muscle forces, normalized to maximum isometric force
    obs_body_space[:,[21 + x for x in list(range(0, 33, 3)) + list(range(44, 77, 3))]] = np.array([[0, 2]]).transpose() # (r, l) muscle lengths, normalized to optimal length
    obs_body_space[:,[22 + x for x in list(range(0, 33, 3)) + list(range(44, 77, 3))]] = np.array([[-20, 20]]).transpose() # (r, l) muscle velocities, normalized to optimal length per second

    def get_model_key(self):
        return self.model

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        if difficulty == 0:
            self.time_limit = 1000
        if difficulty == 1:
            self.time_limit = 1000
        if difficulty == 2:
            self.time_limit = 1000
        self.spec.timestep_limit = self.time_limit    

    def __init__(self, visualize=True, integrator_accuracy=5e-5, difficulty=2, seed=0, report=None):
        if difficulty not in [0, 1, 2]:
            raise ValueError("difficulty level should be in [0, 1, 2].")
        self.model_paths = {}
        self.model_paths['3D'] = os.path.join(os.path.dirname(__file__), '../models/gait14dof22musc_20170320.osim')
        self.model_paths['2D'] = os.path.join(os.path.dirname(__file__), '../models/gait14dof22musc_planar_20170320.osim')
        self.model_path = self.model_paths[self.get_model_key()]
        super(L2M2019Env, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy)

        self.Fmax = {}
        self.lopt = {}
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            self.Fmax[leg] = {}
            self.lopt[leg] = {}
            for MUS, mus in zip(    ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA'],
                                    ['abd', 'add', 'iliopsoas', 'glut_max', 'hamstrings', 'rect_fem', 'vasti', 'bifemsh', 'gastroc', 'soleus', 'tib_ant']):
                muscle = self.osim_model.muscleSet.get('{}_{}'.format(mus,side))
                Fmax = muscle.getMaxIsometricForce()
                lopt = muscle.getOptimalFiberLength()
                self.Fmax[leg][MUS] = muscle.getMaxIsometricForce()
                self.lopt[leg][MUS] = muscle.getOptimalFiberLength()

        self.set_difficulty(difficulty)

        if report:
            bufsize = 0
            self.observations_file = open('%s-obs.csv' % (report,),'w', bufsize)
            self.actions_file = open('%s-act.csv' % (report,),'w', bufsize)
            self.get_headers()

        # create target velocity field
        from envs.target import VTgtField
        self.vtgt = VTgtField(visualize=visualize, version=self.difficulty, dt=self.osim_model.stepsize)
        self.obs_vtgt_space = self.vtgt.vtgt_space

    def reset(self, project=True, seed=None, init_pose=None, obs_as_dict=True):
        self.t = 0
        self.init_reward()
        self.vtgt.reset(version=self.difficulty, seed=seed)

        # initialize state
        self.osim_model.state = self.osim_model.model.initializeState()
        if init_pose is None:
            init_pose = self.INIT_POSE
        state = self.osim_model.get_state()
        QQ = state.getQ()
        QQDot = state.getQDot()
        for i in range(17):
            QQDot[i] = 0
        QQ[3] = 0 # x: (+) forward
        QQ[5] = 0 # z: (+) right
        QQ[1] = 0*np.pi/180 # roll
        QQ[2] = 0*np.pi/180 # yaw
        QQDot[3] = init_pose[0] # forward speed
        QQDot[5] = init_pose[1] # forward speed
        QQ[4] = init_pose[2] # pelvis height
        QQ[0] = -init_pose[3] # trunk lean: (+) backward
        QQ[7] = -init_pose[4] # right hip abduct
        QQ[6] = -init_pose[5] # right hip flex
        QQ[13] = init_pose[6] # right knee extend
        QQ[15] = -init_pose[7] # right ankle flex
        QQ[10] = -init_pose[8] # left hip adduct
        QQ[9] = -init_pose[9] # left hip flex
        QQ[14] = init_pose[10] # left knee extend
        QQ[16] = -init_pose[11] # left ankle flex

        state.setQ(QQ)
        state.setU(QQDot)
        self.osim_model.set_state(state)
        self.osim_model.model.equilibrateMuscles(self.osim_model.state)

        self.osim_model.state.setTime(0)
        self.osim_model.istep = 0

        self.osim_model.reset_manager()

        d = super(L2M2019Env, self).get_state_desc()
        pose = np.array([d['body_pos']['pelvis'][0], -d['body_pos']['pelvis'][2], d['joint_pos']['ground_pelvis'][2]])
        self.v_tgt_field, self.flag_new_v_tgt_field = self.vtgt.update(pose)
        
        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return self.get_observation_dict()
        return self.get_observation()

    def load_model(self, model_path = None):
        super(L2M2019Env, self).load_model(model_path)
        observation_space = np.concatenate((self.obs_vtgt_space, self.obs_body_space), axis=1)
        self.observation_space = convert_to_gym(observation_space)

    def step(self, action, project=True, obs_as_dict=True):
        action_mapped = [action[i] for i in self.act2mus]
        observation, reward, done, info = super(L2M2019Env, self).step(action_mapped, project=project, obs_as_dict=obs_as_dict)
        self.t += self.osim_model.stepsize
        self.update_footstep()

        d = super(L2M2019Env, self).get_state_desc()
        self.pose = np.array([d['body_pos']['pelvis'][0], -d['body_pos']['pelvis'][2], d['joint_pos']['ground_pelvis'][2]])
        self.v_tgt_field, self.flag_new_v_tgt_field = self.vtgt.update(self.pose)

        return observation, reward, done, info

    def change_model(self, model='3D', difficulty=2, seed=0):
        if self.model != model:
            self.model = model
            self.load_model(self.model_paths[self.get_model_key()])
        self.set_difficulty(difficulty)
    
    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc['body_pos']['pelvis'][1] < 0.6

    def update_footstep(self):
        state_desc = self.get_state_desc()

        # update contact
        r_contact = True if state_desc['forces']['foot_r'][1] < -0.05*(self.MASS*self.G) else False
        l_contact = True if state_desc['forces']['foot_l'][1] < -0.05*(self.MASS*self.G) else False

        self.footstep['new'] = False
        if (not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def get_observation_dict(self):
        state_desc = self.get_state_desc()

        obs_dict = {}

        obs_dict['v_tgt_field'] = state_desc['v_tgt_field']

        # pelvis state (in local frame)
        obs_dict['pelvis'] = {}
        obs_dict['pelvis']['height'] = state_desc['body_pos']['pelvis'][1]
        obs_dict['pelvis']['pitch'] = -state_desc['joint_pos']['ground_pelvis'][0] # (+) pitching forward
        obs_dict['pelvis']['roll'] = state_desc['joint_pos']['ground_pelvis'][1] # (+) rolling around the forward axis (to the right)
        yaw = state_desc['joint_pos']['ground_pelvis'][2]
        dx_local, dy_local = rotate_frame(  state_desc['body_vel']['pelvis'][0],
                                            state_desc['body_vel']['pelvis'][2],
                                            yaw)
        dz_local = state_desc['body_vel']['pelvis'][1]
        obs_dict['pelvis']['vel'] = [   dx_local, # (+) forward
                                        -dy_local, # (+) leftward
                                        dz_local, # (+) upward
                                        -state_desc['joint_vel']['ground_pelvis'][0], # (+) pitch angular velocity
                                        state_desc['joint_vel']['ground_pelvis'][1], # (+) roll angular velocity
                                        state_desc['joint_vel']['ground_pelvis'][2]] # (+) yaw angular velocity

        # leg state
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            obs_dict[leg] = {}
            grf = [ f/(self.MASS*self.G) for f in state_desc['forces']['foot_{}'.format(side)][0:3] ] # forces normalized by bodyweight
            grm = [ m/(self.MASS*self.G) for m in state_desc['forces']['foot_{}'.format(side)][3:6] ] # forces normalized by bodyweight
            grfx_local, grfy_local = rotate_frame(-grf[0], -grf[2], yaw)
            if leg == 'r_leg':
                obs_dict[leg]['ground_reaction_forces'] = [ grfx_local, # (+) forward
                                                            grfy_local, # (+) lateral (rightward)
                                                            -grf[1]] # (+) upward
            if leg == 'l_leg':
                obs_dict[leg]['ground_reaction_forces'] = [ grfx_local, # (+) forward
                                                            -grfy_local, # (+) lateral (leftward)
                                                            -grf[1]] # (+) upward

            # joint angles
            obs_dict[leg]['joint'] = {}
            obs_dict[leg]['joint']['hip_abd'] = -state_desc['joint_pos']['hip_{}'.format(side)][1] # (+) hip abduction
            obs_dict[leg]['joint']['hip'] = -state_desc['joint_pos']['hip_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['joint']['knee'] = state_desc['joint_pos']['knee_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['joint']['ankle'] = -state_desc['joint_pos']['ankle_{}'.format(side)][0] # (+) extension
            # joint angular velocities
            obs_dict[leg]['d_joint'] = {}
            obs_dict[leg]['d_joint']['hip_abd'] = -state_desc['joint_vel']['hip_{}'.format(side)][1] # (+) hip abduction
            obs_dict[leg]['d_joint']['hip'] = -state_desc['joint_vel']['hip_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['d_joint']['knee'] = state_desc['joint_vel']['knee_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['d_joint']['ankle'] = -state_desc['joint_vel']['ankle_{}'.format(side)][0] # (+) extension

            # muscles
            for MUS, mus in zip(    ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA'],
                                    ['abd', 'add', 'iliopsoas', 'glut_max', 'hamstrings', 'rect_fem', 'vasti', 'bifemsh', 'gastroc', 'soleus', 'tib_ant']):
                obs_dict[leg][MUS] = {}
                obs_dict[leg][MUS]['f'] = state_desc['muscles']['{}_{}'.format(mus,side)]['fiber_force']/self.Fmax[leg][MUS]
                obs_dict[leg][MUS]['l'] = state_desc['muscles']['{}_{}'.format(mus,side)]['fiber_length']/self.lopt[leg][MUS]
                obs_dict[leg][MUS]['v'] = state_desc['muscles']['{}_{}'.format(mus,side)]['fiber_velocity']/self.lopt[leg][MUS]

        return obs_dict

    ## Values in the observation vector
    # 'vtgt_field': vtgt vectors in body frame (2*11*11 = 242 values)
    # 'pelvis': height, pitch, roll, 6 vel (9 values)
    # for each 'r_leg' and 'l_leg' (*2)
    #   'ground_reaction_forces' (3 values)
    #   'joint' (4 values)
    #   'd_joint' (4 values)
    #   for each of the eleven muscles (*11)
    #       normalized 'f', 'l', 'v' (3 values)
    # 242 + 9 + 2*(3 + 4 + 4 + 11*3) = 339
    def get_observation(self):
        obs_dict = self.get_observation_dict()

        # Augmented environment from the L2R challenge
        res = []

        # target velocity field (in body frame)
        v_tgt = np.ndarray.flatten(obs_dict['v_tgt_field'])
        res += v_tgt.tolist()

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][1]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][2]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])
        return res

    def get_observation_space_size(self):
        return 339
        
    def get_state_desc(self):
        d = super(L2M2019Env, self).get_state_desc()
        #state_desc['joint_pos']
        #state_desc['joint_vel']
        #state_desc['joint_acc']
        #state_desc['body_pos']
        #state_desc['body_vel']
        #state_desc['body_acc']
        #state_desc['body_pos_rot']
        #state_desc['body_vel_rot']
        #state_desc['body_acc_rot']
        #state_desc['forces']
        #state_desc['muscles']
        #state_desc['markers']
        #state_desc['misc']
        if self.difficulty in [0, 1, 2]:
            d['v_tgt_field'] = self.v_tgt_field # shape: (2, 11, 11)
        else:
            raise ValueError("difficulty level should be in [0, 1, 2].")
        return d

    def init_reward(self):
        self.init_reward_1()

    def init_reward_1(self):
        self.d_reward = {}

        self.d_reward['weight'] = {}
        self.d_reward['weight']['footstep'] = 10
        self.d_reward['weight']['effort'] = 1
        self.d_reward['weight']['v_tgt'] = 1

        self.d_reward['alive'] = 0.1
        self.d_reward['effort'] = 0

        self.d_reward['footstep'] = {}
        self.d_reward['footstep']['effort'] = 0
        self.d_reward['footstep']['del_t'] = 0
        self.d_reward['footstep']['del_v'] = 0

    def get_reward(self):
        return self.get_reward_1()

    def get_reward_1(self):
        state_desc = self.get_state_desc()
        if not self.get_prev_state_desc():
            return 0

        reward = 0
        dt = self.osim_model.stepsize

        # alive reward
        # should be large enough to search for 'success' solutions (alive to the end) first
        reward += self.d_reward['alive']

        # effort ~ muscle fatigue ~ (muscle activation)^2 
        ACT2 = 0
        for muscle in sorted(state_desc['muscles'].keys()):
            ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
        self.d_reward['effort'] += ACT2*dt
        self.d_reward['footstep']['effort'] += ACT2*dt

        self.d_reward['footstep']['del_t'] += dt

        # reward from velocity (penalize from deviating from v_tgt)

        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        v_tgt = self.vtgt.get_vtgt(p_body).T

        self.d_reward['footstep']['del_v'] += (v_body - v_tgt)*dt

        # footstep reward (when made a new step)
        if self.footstep['new']:
            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
            reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

            # deviation from target velocity
            # the average velocity a step (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking
            #reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
            reward_footstep_v = -self.d_reward['weight']['v_tgt']*np.linalg.norm(self.d_reward['footstep']['del_v'])/self.LENGTH0

            # panalize effort
            reward_footstep_e = -self.d_reward['weight']['effort']*self.d_reward['footstep']['effort']

            self.d_reward['footstep']['del_t'] = 0
            self.d_reward['footstep']['del_v'] = 0
            self.d_reward['footstep']['effort'] = 0

            reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e

        # success bonus
        if not self.is_done() and (self.osim_model.istep >= self.spec.timestep_limit): #and self.failure_mode is 'success':
            # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
            reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
            reward += reward_footstep_0 + 100

        return reward


def rotate_frame(x, y, theta):
    x_rot = np.cos(theta)*x - np.sin(theta)*y
    y_rot = np.sin(theta)*x + np.cos(theta)*y
    return x_rot, y_rot
