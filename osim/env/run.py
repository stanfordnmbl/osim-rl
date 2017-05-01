import opensim
import math
import numpy as np
import os
import random
import string
from .osim import OsimEnv

def generate_env(difficulty, seed):
    if seed:
        np.random.seed(seed)
    num_obstacles = np.random.poisson(difficulty, 1)

    xs = np.random.uniform(0.0, 20.0, num_obstacles)
    ys = np.random.uniform(-0.5, 1, num_obstacles)
    rs = np.random.exponential(0.1, num_obstacles)

    ys = map(lambda (x,y): x*y, zip(ys, rs))

    return zip(xs,ys,rs)
    
class RunEnv(OsimEnv):
    ninput = 31
    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')
    ligamentSet = []

    def __init__(self, visualize = True, noutput = None):
        super(RunEnv, self).__init__(visualize = False, noutput = noutput)
        self.create_obstacles()
        self.osim_model.model.setUseVisualizer(visualize)
        self.osim_model.model.initSystem()

    def setup(self, difficulty, seed=None):
        # create the new env
        self.clear_obstacles()
        obstacles = generate_env(difficulty, seed)
        for x,y,r in obstacles:
            self.add_obstacle(x,y,r)
        state = self.osim_model.model.initializeState()
        self.osim_model.model.equilibrateMuscles(state)

    def reset(self):
        self.last_state = [0] * self.ninput
        self.current_state = [0] * self.ninput
        return super(RunEnv, self).reset()

    def getHead(self):
        return self.osim_model.bodies[2].getTransformInGround(self.osim_model.state).p()

    def getFootL(self):
        return self.osim_model.bodies[0].getTransformInGround(self.osim_model.state).p()

    def getFootR(self):
        return self.osim_model.bodies[1].getTransformInGround(self.osim_model.state).p()

    def getPelvis(self):
        return self.osim_model.bodies[3].getTransformInGround(self.osim_model.state).p()

    def compute_reward(self):
        lig_pen = 0
        for lig in self.ligamentSet:
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        delta = self.current_state[2] - self.last_state[2]

        return delta - lig_pen * 0.0001

    def is_pelvis_too_low(self):
        y = self.osim_model.joints[0].getCoordinate(2).getValue(self.osim_model.state)
        return (y < 0.65)
    
    def is_done(self):
        return self.is_pelvis_too_low()

    def configure(self):
        super(RunEnv, self).configure()

        self.osim_model.joints.append(opensim.PlanarJoint.safeDownCast(self.osim_model.jointSet.get(0)))

        self.osim_model.joints.append(opensim.PinJoint.safeDownCast(self.osim_model.jointSet.get(1)))
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(2)))
        self.osim_model.joints.append(opensim.PinJoint.safeDownCast(self.osim_model.jointSet.get(3)))

        self.osim_model.joints.append(opensim.PinJoint.safeDownCast(self.osim_model.jointSet.get(6)))
        self.osim_model.joints.append(opensim.CustomJoint.safeDownCast(self.osim_model.jointSet.get(7)))
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

        # Get ligaments
        self.forceSet = self.osim_model.model.getForceSet()
        for j in range(20, 26):
            self.ligamentSet.append(opensim.CoordinateLimitForce.safeDownCast(self.forceSet.get(j)))

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

    obstacles = []
    num_obstacles = 0
    
    def create_obstacles(self):
        x = 0.05
        y = -0.1
        r = 0.01
        for i in range(20):
            name = i.__str__()
            blockos = opensim.Body(name + '-block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
            pj = opensim.PinJoint(name + '-pin',
                                  self.osim_model.model.getGround(), # PhysicalFrame
                                  opensim.Vec3(x, y, 0),
                                  opensim.Vec3(0, 0, 0),
                                  blockos, # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0))

            bodyGeometry = opensim.Ellipsoid(r, r, r)
            bodyGeometry.setColor(opensim.Orange)
            blockos.attachGeometry(bodyGeometry)

            self.osim_model.model.addComponent(pj)
            self.osim_model.model.addComponent(blockos)

            block = opensim.ContactSphere(r, opensim.Vec3(0,0,0), blockos)
            block.setName(name + '-contact')
            self.osim_model.model.addContactGeometry(block)

            force = opensim.HuntCrossleyForce()
            
            force.addGeometry(name + '-contact')
            force.addGeometry("r_heel")
            force.addGeometry("l_heel")
            force.addGeometry("r_toe")
            force.addGeometry("l_toe")
        
            force.setStiffness(1.0e6/r)
            force.setDissipation(1e-5)
            force.setStaticFriction(0.0)
            force.setDynamicFriction(0.0)
            force.setViscousFriction(0.0)

            self.obstacles.append({
                'joint': pj,
                'force': force,
                'ball': bodyGeometry,
                'contact': block,
            })

            self.osim_model.model.addForce(force);
        self.clear_obstacles()

    def add_obstacle(self, x, y, r):
        # set obstacle number num_obstacles
        # self.osim_model.model
        newloc = opensim.Vec3(x, y, 0)
#        print(self.obstacles[self.num_obstacles]["joint"].setLocationInParent)
#        self.obstacles[self.num_obstacles]["joint"].setLocationInParent(newloc)
        self.obstacles[self.num_obstacles]["ball"].setEllipsoidParams(r,r,r)
        self.obstacles[self.num_obstacles]["contact"].setRadius(r)
        self.obstacles[self.num_obstacles]["force"].setStiffness(1.0e6/r)
        


        self.num_obstacles += 1
        pass

    def clear_obstacles(self):
        for i in range(20):
            # Set to 0
            pass
        self.num_obstacles = 0
        
