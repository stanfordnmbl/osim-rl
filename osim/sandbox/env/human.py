import opensim
import math
import numpy as np
import os
from .osim import OsimEnv

class GaitEnv(OsimEnv):
    ninput = 31
    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')

    def reset(self):
        self.last_state = [0] * self.ninput
        self.current_state = [0] * self.ninput
        return super(GaitEnv, self).reset()

    def getHead(self):
        return self.osim_model.bodies[2].getTransformInGround(self.osim_model.state).p()

    def getFootL(self):
        return self.osim_model.bodies[0].getTransformInGround(self.osim_model.state).p()

    def getFootR(self):
        return self.osim_model.bodies[1].getTransformInGround(self.osim_model.state).p()

    def getPelvis(self):
        return self.osim_model.bodies[3].getTransformInGround(self.osim_model.state).p()

    def compute_reward(self):        
        delta = self.current_state[2] - self.last_state[2]

        return delta

    def is_pelvis_too_low(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y < 0.7)
    
    def is_done(self):
        return self.is_pelvis_too_low()

    def __init__(self, visualize = True, noutput = None):
        super(GaitEnv, self).__init__(visualize = visualize, noutput = noutput)

    def configure(self):
        super(GaitEnv, self).configure()

        self.osim_model.joints.append(opensim.PlanarJoint.safeDownCast(self.osim_model.jointSet.get(0))) # PELVIS

        self.osim_model.joints.append(opensim.PinJoint.safeDownCast(self.osim_model.jointSet.get(1)))
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(2))) # 4
        self.osim_model.joints.append(opensim.PinJoint.safeDownCast(self.osim_model.jointSet.get(3)))    # 7
        # self.osim_model.joints.append(opensim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(4)))
        # self.osim_model.joints.append(opensim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(5)))

        self.osim_model.joints.append(opensim.PinJoint.safeDownCast(self.osim_model.jointSet.get(6)))    # 2
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(7))) # 5
        self.osim_model.joints.append(opensim.PinJoint.safeDownCast(self.osim_model.jointSet.get(8)))
        # self.osim_model.joints.append(opensim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(9)))
        # self.osim_model.joints.append(opensim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(10)))

        # self.osim_model.joints.append(opensim.PinJoint.safeDownCast(self.osim_model.jointSet.get(11)))
        # self.osim_model.joints.append(opensim.WeldJoint.safeDownCast(self.osim_model.jointSet.get(12)))

        for i in range(13):
            print(self.osim_model.bodySet.get(i).getName())

        self.osim_model.bodies.append(self.osim_model.bodySet.get(5))
        self.osim_model.bodies.append(self.osim_model.bodySet.get(10))
        self.osim_model.bodies.append(self.osim_model.bodySet.get(12))
        self.osim_model.bodies.append(self.osim_model.bodySet.get(0))

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

        invars[21] = vel[0]
        invars[22] = vel[1]

        posH = self.getHead()
        posP = self.getPelvis()
        self.currentL = self.getFootL()
        self.currentR = self.getFootR()

        invars[23] = posH[0]
        invars[24] = posH[1]

        invars[25] = posP[0]
        invars[26] = posP[1]

        invars[27] = self.currentL[0]
        invars[28] = self.currentL[1]

        invars[29] = self.currentR[0]
        invars[30] = self.currentR[1]


        self.current_state = invars
        
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
