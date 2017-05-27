import opensim
import math
import numpy as np
import os
import random
import string
from .osim import OsimEnv

def generate_env(difficulty, seed, max_obstacles):
    if seed:
        np.random.seed(seed)

    # obstacles
    num_obstacles = max(0, min(int(difficulty/3), max_obstacles)) # min(np.random.poisson(difficulty, 1), max_obstacles)

    xs = np.random.uniform(2, 10.0, num_obstacles)
    ys = np.random.uniform(-0.5, 0.25, num_obstacles)
    rs = [0.05 + r for r in np.random.exponential(0.01*difficulty, num_obstacles)]

    ys = map(lambda (x,y): x*y, zip(ys, rs))

    # muscle strength
    rpsoas = 1.1 - np.random.normal(0, difficulty / 20.0)
    lpsoas = 1.1 - np.random.normal(0, difficulty / 20.0)
    rpsoas = max(0.1, rpsoas)
    lpsoas = max(0.1, lpsoas)
    muscles = [1] * 18

    # modify only psoas
    muscles[3] = rpsoas
    muscles[11] = lpsoas

    obstacles = zip(xs,ys,rs)
    obstacles.sort()

    return {
        'muscles': muscles,
        'obstacles': obstacles
    }
    
class RunEnv(OsimEnv):
    obstacles = []
    num_obstacles = 0
    max_obstacles = None

    model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')
    ligamentSet = []
    verbose = False
    pelvis = None
    env_desc = {"obstacles": [], "muscles": [1]*18}

    ninput = 41
    noutput = 18

    def __init__(self, visualize = True, noutput = None, max_obstacles = 3):
        self.max_obstacles = max_obstacles
        super(RunEnv, self).__init__(visualize = False, noutput = noutput)
        self.osim_model.model.setUseVisualizer(visualize)
        self.create_obstacles()
        state = self.osim_model.model.initSystem()

        if visualize:
            manager = opensim.Manager(self.osim_model.model)
            manager.setInitialTime(-0.00001)
            manager.setFinalTime(0.0)
            manager.integrate(state)

    def setup(self, difficulty, seed=None):
        # create the new env
        # set up obstacles
        self.env_desc = generate_env(difficulty, seed, self.max_obstacles)
        
        self.clear_obstacles(self.osim_model.state)
        for x,y,r in self.env_desc['obstacles']:
            self.add_obstacle(self.osim_model.state,x,y,r)

        # set up muscle strength
        self.osim_model.set_strength(self.env_desc['muscles'])

    def reset(self, difficulty=0, seed=None):
        self.last_state = super(RunEnv, self).reset()
        self.setup(difficulty, seed)
        self.current_state = self.last_state
        return self.get_observation()

    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        for lig in self.ligamentSet:
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        # Get the pelvis X delta
        delta_x = self.current_state[1] - self.last_state[1]

        return delta_x - math.sqrt(lig_pen) * 0.001

    def is_pelvis_too_low(self):
        return (self.current_state[2] < 0.65)
    
    def is_done(self):
        return self.is_pelvis_too_low()

    def configure(self):
        super(RunEnv, self).configure()

        if self.verbose:
            print("JOINTS")
            for i in range(11):
                print(i,self.osim_model.jointSet.get(i).getName())
            print("\nBODIES")
            for i in range(13):
                print(i,self.osim_model.bodySet.get(i).getName())
            print("\nMUSCLES")
            for i in range(18):
                print(i,self.osim_model.muscleSet.get(i).getName())
            print("\nFORCES")
            for i in range(26):
                print(i,self.osim_model.forceSet.get(i).getName())
            print("")


        # The only joint that has to be cast
        self.pelvis = opensim.PlanarJoint.safeDownCast(self.osim_model.get_joint("ground_pelvis"))

        # Get ligaments
        self.forceSet = self.osim_model.model.getForceSet()
        for j in range(20, 26):
            self.ligamentSet.append(opensim.CoordinateLimitForce.safeDownCast(self.forceSet.get(j)))

    def next_obstacle(self):
        obstacles = self.env_desc['obstacles']
        x = self.pelvis.getCoordinate(1).getValue(self.osim_model.state)
        for obstacle in obstacles:
            if obstacle[0] + obstacle[2] < x:
                continue
            else:
                return map(lambda (x,y): x-y, zip(obstacle, [x,0,0]))
        return [100,0,0]
        

    def get_observation(self):
        bodies = ['head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']

        pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
        pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]

        jnts = ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
        joint_angles = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in range(6)]
        joint_vel = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in range(6)]

        mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]  
        mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

        body_transforms = [[self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] for i in range(2)] for body in bodies]

        muscles = [ self.env_desc['muscles'][3], self.env_desc['muscles'][11] ]
    
        # see the next obstacle
        obstacle = self.next_obstacle()

        self.current_state = pelvis_pos + pelvis_vel + joint_angles + joint_vel + mass_pos + mass_vel + reduce(lambda x,y: x+y, body_transforms) + muscles + obstacle
        return self.current_state

    def create_obstacles(self):
        x = 0
        y = 0
        r = 0.1
        for i in range(self.max_obstacles):
            name = i.__str__()
            blockos = opensim.Body(name + '-block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
            pj = opensim.PlanarJoint(name + '-pin',
                                  self.osim_model.model.getGround(), # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0),
                                  blockos, # PhysicalFrame
                                  opensim.Vec3(0, 0, 0),
                                  opensim.Vec3(0, 0, 0))

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
                'contact': block,
            })

            self.osim_model.model.addForce(force);

    def clear_obstacles(self, state):
        for j in range(0, self.max_obstacles):
            joint = self.obstacles[j]["joint"]
            # joint.getCoordinate(1).setValue(state, 0)
            # joint.getCoordinate(2).setValue(state, 0)
            # self.obstacles[j]["contact"].setRadius(0.1)
            # for i in range(3):
            #     joint.getCoordinate(i).setLocked(state, True)

        self.num_obstacles = 0
        pass
        
    def add_obstacle(self, state, x, y, r):
        # set obstacle number num_obstacles
        self.obstacles[self.num_obstacles]["contact"].setRadius(r)
        self.obstacles[self.num_obstacles]["force"].setStiffness(1.0e6/r)

        joint = self.obstacles[self.num_obstacles]["joint"]
        newpos = [x,y] 
        for i in range(2):
            joint.getCoordinate(1 + i).setLocked(state, False)
            joint.getCoordinate(1 + i).setValue(state, newpos[i], False)
            joint.getCoordinate(1 + i).setLocked(state, True)

        self.num_obstacles += 1
        pass
