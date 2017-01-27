import opensim
import math
import numpy as np
import os
import random
from .osim import OsimEnv

class ArmEnv(OsimEnv):
    ninput = 14
    model_path = os.path.join(os.path.dirname(__file__), '../models/arm2dof6musc.osim')

    def __init__(self, visualize = False):
        self.iepisode = 0
        self.shoulder = 0.0
        self.elbow = 0.0
        super(ArmEnv, self).__init__(visualize = visualize)

    def configure(self):
        super(ArmEnv, self).configure()
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(0)))
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(1)))

    def new_target(self):
        self.shoulder = random.uniform(-1.2,0.3)
        self.elbow = random.uniform(-1.0,0)

    def reset(self):
        self.new_target()
        return super(ArmEnv, self).reset()

    def compute_reward(self):
        obs = self.get_observation()
        pos = (self.angular_dist(obs[2],self.shoulder) + self.angular_dist(obs[3],self.elbow))
        speed = 0 #(obs[4]**2 + obs[5]**2) / 200.0
        return - pos - speed

    def get_observation(self):
        invars = np.array([0] * self.ninput, dtype='f')

        invars[0] = self.shoulder
        invars[1] = self.elbow
        
        invars[2] = self.osim_model.joints[0].getCoordinate(0).getValue(self.osim_model.state)
        invars[3] = self.osim_model.joints[1].getCoordinate(0).getValue(self.osim_model.state)

        invars[4] = self.osim_model.joints[0].getCoordinate(0).getSpeedValue(self.osim_model.state)
        invars[5] = self.osim_model.joints[1].getCoordinate(0).getSpeedValue(self.osim_model.state)

        invars[6] = self.sanitify(self.osim_model.joints[0].getCoordinate(0).getAccelerationValue(self.osim_model.state))
        invars[7] = self.sanitify(self.osim_model.joints[1].getCoordinate(0).getAccelerationValue(self.osim_model.state))

        pos = self.osim_model.model.calcMassCenterPosition(self.osim_model.state)
        vel = self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)
        
        invars[8] = pos[0]
        invars[9] = pos[1]
        invars[10] = pos[2]

        invars[11] = vel[0]
        invars[12] = vel[1]
        invars[13] = vel[2]

        return invars

