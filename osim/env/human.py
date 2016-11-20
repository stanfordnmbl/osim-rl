import opensim as osim
import math
import numpy as np
import os
from env.osim import OsimEnv

class GaitEnv(OsimEnv):
    ninput = 25
    model_path = os.path.join(os.path.dirname(__file__), '../../models/gait9dof18musc.osim')

    def compute_reward(self):
        obs = self.get_observation()
        return self.joints[0].getCoordinate(1).getValue(self.state)

    def is_head_too_low(self):
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        return (y < 0.5)
    
    def is_done(self):
        return self.is_head_too_low()

    def __init__(self, visualize = True, noutput = None):
        super(GaitEnv, self).__init__(visualize = visualize, noutput = noutput)

        self.head = self.bodySet.get(12)

        self.joints.append(osim.PlanarJoint.safeDownCast(self.jointSet.get(0))) # PELVIS

        self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(1)))
        self.joints.append(osim.CustomJoint.safeDownCast(self.jointSet.get(2))) # 4
        self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(3)))    # 7
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(4)))
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(5)))

        self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(6)))    # 2
        self.joints.append(osim.CustomJoint.safeDownCast(self.jointSet.get(7))) # 5
        self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(8)))
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(9)))
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(10)))

        # self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(11)))
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(12)))
        
        for i in range(18):
            print(self.muscleSet.get(i).getName())
        

        self.reset()

    def get_observation(self):
        invars = np.array([0] * self.ninput, dtype='f')

        invars[0] = 0.0

        invars[1] = self.joints[0].getCoordinate(0).getValue(self.state)
        invars[2] = self.joints[0].getCoordinate(1).getValue(self.state)
        invars[3] = self.joints[0].getCoordinate(2).getValue(self.state)

        invars[4] = self.joints[0].getCoordinate(0).getSpeedValue(self.state)
        invars[5] = self.joints[0].getCoordinate(1).getSpeedValue(self.state)
        invars[6] = self.joints[0].getCoordinate(2).getSpeedValue(self.state)

        for i in range(6):
            invars[7+i] = self.joints[1+i].getCoordinate(0).getValue(self.state)
        for i in range(6):
            invars[13+i] = self.joints[1+i].getCoordinate(0).getSpeedValue(self.state)

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        
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
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        x = self.joints[0].getCoordinate(1).getValue(self.state)

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        acc = self.model.calcMassCenterAcceleration(self.state)

        a = abs(acc[0])**2 + abs(acc[1])**2 + abs(acc[2])**2
        v = abs(vel[0])**2 + abs(vel[1])**2 + abs(vel[2])**2
        rew = 50.0 - min(a,10.0) - min(v,40.0)

        return rew / 50.0

class HopEnv(GaitEnv):
    def __init__(self, visualize = True):
        self.model_path = os.path.join(os.path.dirname(__file__), '../../models/hop8dof9musc.osim')
        super(HopEnv, self).__init__(visualize = visualize, noutput = 9)

    def compute_reward(self):
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        return (y) ** 3

    def is_head_too_low(self):
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        return (y < 0.4)

    def activate_muscles(self, action):
        for j in range(9):
            muscle = self.muscleSet.get(j)
            muscle.setActivation(self.state, action[j])
            muscle = self.muscleSet.get(j + 9)
            muscle.setActivation(self.state, action[j])

class CrouchEnv(HopEnv):
    def compute_reward(self):
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        return 1.0 - (y-0.5) ** 3

    def is_head_too_low(self):
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        return (y < 0.25)
