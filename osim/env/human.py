import math
import numpy as np
import os
from .osim import OsimEnv

class GaitEnv(OsimEnv):
    ninput = 25
    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')

    def compute_reward(self):
        obs = self.get_observation()
        return self.osim_model.joints[0].getCoordinate(1).getValue(self.osim_model.state)

    def is_head_too_low(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y < 0.5)
    
    def is_done(self):
        return self.is_head_too_low()

    def __init__(self, visualize = True, noutput = None):
        super(GaitEnv, self).__init__(visualize = visualize, noutput = noutput)

    def configure(self):
        super(GaitEnv, self).configure()

        self.osim_model.joints.append(osim.PlanarJoint.safeDownCast(self.osim_model.jointSet.get(0))) # PELVIS

        self.osim_model.joints.append(osim.PinJoint.safeDownCast(self.osim_model.jointSet.get(1)))
        self.osim_model.joints.append(osim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(2))) # 4
        self.osim_model.joints.append(osim.PinJoint.safeDownCast(self.osim_model.jointSet.get(3)))    # 7
        # self.osim_model.joints.append(osim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(4)))
        # self.osim_model.joints.append(osim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(5)))

        self.osim_model.joints.append(osim.PinJoint.safeDownCast(self.osim_model.jointSet.get(6)))    # 2
        self.osim_model.joints.append(osim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(7))) # 5
        self.osim_model.joints.append(osim.PinJoint.safeDownCast(self.osim_model.jointSet.get(8)))
        # self.osim_model.joints.append(osim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(9)))
        # self.osim_model.joints.append(osim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(10)))

        # self.osim_model.joints.append(osim.PinJoint.safeDownCast(self.osim_model.jointSet.get(11)))
        # self.osim_model.joints.append(osim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(12)))
        
        for i in range(18):
            print(self.osim_model.muscleSet.get(i).getName())

    def get_observation(self):
        invars = np.array([0] * self.ninput, dtype='f')

        invars[0] = 0.0

        invars[1] = self.osim_model.joints[0].getCoordinate(0).getValue(self.osim_model.state)
        invars[2] = self.osim_model.joints[0].getCoordinate(1).getValue(self.osim_model.state)
        invars[3] = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)

        invars[4] = self.osim_model.joints[0].getCoordinate(0).getSpeedValue(self.osim_model.state)
        invars[5] = self.osim_model.joints[0].getCoordinate(1).getSpeedValue(self.osim_model.state)
        invars[6] = self.osim_model.joints[0].getCoordinate(2).getSpeedValue(self.osim_model.state)

        for i in range(6):
            invars[7+i] = self.osim_model.joints[1+i].getCoordinate(0).getValue(self.osim_model.state)
        for i in range(6):
            invars[13+i] = self.osim_model.joints[1+i].getCoordinate(0).getSpeedValue(self.osim_model.state)

        pos = self.osim_model.model.calcMassCenterPosition(self.osim_model.state)
        vel = self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)
        
        invars[19] = pos[0]
        invars[20] = pos[1]
        invars[21] = pos[2]

        invars[22] = vel[0]
        invars[23] = vel[1]
        invars[24] = vel[2]

        # for i in range(0,self.ninput):
        #     invars[i] = self.sanitify(invars[i])

        return invars

class StandEnv(GaitEnv):
    def compute_reward(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        x = self.osim_model.joints[0].getCoordinate(1).getValue(self.osim_model.state)

        pos = self.osim_model.model.calcMassCenterPosition(self.osim_model.state)
        vel = self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)
        acc = self.osim_model.model.calcMassCenterAcceleration(self.osim_model.state)

        a = abs(acc[0])**2 + abs(acc[1])**2 + abs(acc[2])**2
        v = abs(vel[0])**2 + abs(vel[1])**2 + abs(vel[2])**2
        rew = 50.0 - min(a,10.0) - min(v,40.0)

        return rew / 50.0

class HopEnv(GaitEnv):
    def __init__(self, visualize = True):
        self.model_path = os.path.join(os.path.dirname(__file__), '../models/hop8dof9musc.osim')
        super(HopEnv, self).__init__(visualize = visualize, noutput = 9)

    def compute_reward(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y) ** 3

    def is_head_too_low(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y < 0.4)

    def activate_muscles(self, action):
        for j in range(9):
            muscle = self.osim_model.muscleSet.get(j)
            muscle.setActivation(self.osim_model.state, action[j])
            muscle = self.osim_model.muscleSet.get(j + 9)
            muscle.setActivation(self.osim_model.state, action[j])

class CrouchEnv(HopEnv):
    def compute_reward(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return 1.0 - (y-0.5) ** 3

    def is_head_too_low(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y < 0.25)
