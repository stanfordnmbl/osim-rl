import opensim
import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym

class Osim(object):
    # Initialize simulation
    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize):
        self.model = opensim.Model(model_path)
        self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()
        
        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.model.addController(self.brain)

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

    def initializeState(self):
        self.state = self.model.initializeState()

    def revert(self, state):
        self.state = state

class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 1000

class OsimEnv(gym.Env):
    stepsize = 0.01
    integration_accuracy = 1e-3
    timestep_limit = 1000
    test = False

    action_space = None
    observation_space = None
    osim_model = None
    istep = 0
    verbose = True

    model_path = ""
    visualize = False
    ninput = 0
    noutput = 0
    last_action = None
    spec = None

    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')    
#    model_path = os.path.join(os.path.dirname(__file__), '../models/MoBL_ARMS_J.osim')    
#    model_path = os.path.join(os.path.dirname(__file__), '../models/MoBL_ARMS_J_Simple_032118.osim')
    
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 50
    }

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['osim_model']
        print ("HERE1")
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)
        self.osim_model = Osim(self.model_path, True)
        self.configure()

    def angular_dist(self, t,s):
        x = (t-s) % (2*math.pi)
        return min(x, 2*math.pi-x)

    def compute_reward(self):
        return 0.0

    def is_done(self):
        return False

    def terminate(self):
        pass

    def __init__(self, visualize = True, noutput = None):
        self.visualize = visualize
        self.osim_model = Osim(self.model_path, self.visualize)

        self.noutput = noutput
        if not noutput:
            self.noutput = self.osim_model.muscleSet.getSize()

        self.spec = Spec()
        self.horizon = self.spec.timestep_limit

        self.configure()

        if not self.action_space:
            self.action_space = ( [0.0] * self.noutput, [1.0] * self.noutput )
        if not self.observation_space:
            self.observation_space = ( [-math.pi] * self.ninput, [math.pi] * self.ninput )
        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)

        state = self.osim_model.model.initSystem()

    def configure(self):
        if self.verbose:
            print("JOINTS")
            for i in range(self.osim_model.jointSet.getSize()):
                print(i,self.osim_model.jointSet.get(i).getName())
            print("\nBODIES")
            for i in range(self.osim_model.bodySet.getSize()):
                print(i,self.osim_model.bodySet.get(i).getName())
            print("\nMUSCLES")
            for i in range(self.osim_model.muscleSet.getSize()):
                print(i,self.osim_model.muscleSet.get(i).getName())
            print("\nFORCES")
            for i in range(self.osim_model.forceSet.getSize()):
                print(i,self.osim_model.forceSet.get(i).getName())
            print("\nMARKERS")
            for i in range(self.osim_model.markerSet.getSize()):
                print(i,self.osim_model.markerSet.get(i).getName())
            print("")

        self.noutput = self.osim_model.muscleSet.getSize()
#        self.ninput =

    def reset(self):
        
        self.istep = 0
        self.osim_model.initializeState()
        return self.get_observation()

    def activate_muscles(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")
        # TODO: Check if actions within [0,1]
        self.last_action = action
            
        brain = opensim.PrescribedController.safeDownCast(self.osim_model.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    def get_observation(self):
        self.osim_model.model.realizeAcceleration(self.osim_model.state)

        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.osim_model.jointSet.getSize()):
            joint = self.osim_model.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.osim_model.state) for i in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.osim_model.state) for i in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.osim_model.state) for i in range(joint.numCoordinates())]

        ## Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.osim_model.bodySet.getSize()):
            body = self.osim_model.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.osim_model.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.osim_model.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.osim_model.state).get(1).get(i) for i in range(3)]
            
            res["body_pos_rot"][name] = [body.getTransformInGround(self.osim_model.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.osim_model.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.osim_model.state).get(0).get(i) for i in range(3)]

        ## Forces
        res["forces"] = {}
        for i in range(self.osim_model.forceSet.getSize()):
            force = self.osim_model.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.osim_model.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.osim_model.muscleSet.getSize()):
            muscle = self.osim_model.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.osim_model.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.osim_model.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.osim_model.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.osim_model.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html 
        
        ## Markers
        res["markers"] = {}
        for i in range(self.osim_model.markerSet.getSize()):
            marker = self.osim_model.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.osim_model.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.osim_model.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.osim_model.state)[i] for i in range(3)]

        return res

    def step(self, action):
        self.activate_muscles(action)

        # Integrate one step
        if self.istep == 0:
            print ("Initializing the model!")
            self.manager = opensim.Manager(self.osim_model.model)
            self.manager.setIntegratorAccuracy(1e-1)
            self.osim_model.state.setTime(self.stepsize * self.istep)
            self.manager.initialize(self.osim_model.state)
            
        try:
            self.osim_model.state = self.manager.integrate(self.stepsize * (self.istep + 1))
        except Exception as e:
            print (e)
            return self.get_observation(), -500, True, {}

        self.istep = self.istep + 1

        res = [ self.get_observation(), self.compute_reward(), self.is_done(), {} ]
        return res

    def render(self, mode='human', close=False):
        return
