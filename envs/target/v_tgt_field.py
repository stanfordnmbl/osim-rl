# Author(s): Seungmoon Song <seungmoon.song@gmail.com>
"""
...
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np
from scipy import interpolate 


class VTgtField(object):
    nn_get = np.array([11, 11]) # vtgt_field_local data is nn_get*nn_get = 121

    # v01: consecutive sinks forward for walking
    v01_res_map = np.array([2, 2])
    v01_rng_xy0 = np.array([[-20, 20], [-20, 20]])
    v01_rng_get = np.array([[-5, 5], [-5, 5]])
    v01_v_amp_rng = np.array([.8, 1.8])
    v01_rng_p_sink_r_th = np.array([[5, 7], [0, 0]])
    v01_r_target = .2

    # v02: consecutive sinks for walking (-180 < th < 180)
    v02_res_map = np.array([2, 2])
    v02_rng_xy0 = np.array([[-20, 20], [-20, 20]])
    v02_rng_get = np.array([[-5, 5], [-5, 5]])
    v02_v_amp_rng = np.array([.8, 1.8])
    v02_rng_p_sink_r_th = np.array([[5, 7], [-90*np.pi/180, 90*np.pi/180]])
    v02_r_target = .2

# -----------------------------------------------------------------------------------------------------------------
    def __init__(self, version=1, pose_agent=np.array([0, 0, 0]), dt=.001):
        # set parameters
        if version is 1:
            self.rng_xy0 = self.v01_rng_xy0
            self.v_amp_rng = self.v01_v_amp_rng
            self.rng_p_sink_r_th = self.v01_rng_p_sink_r_th
            self.r_target = self.v01_r_target
            self.rng_get = self.v01_rng_get
            self.res_map = self.v01_res_map
        elif version is 2:
            self.rng_xy0 = self.v02_rng_xy0
            self.v_amp_rng = self.v02_v_amp_rng
            self.rng_p_sink_r_th = self.v02_rng_p_sink_r_th
            self.r_target = self.v02_r_target
            self.rng_get = self.v02_rng_get
            self.res_map = self.v02_res_map

        self.res_get = np.array([   (self.rng_get[0,1]-self.rng_get[0,0]+1)/self.nn_get[0],
                                    (self.rng_get[1,1]-self.rng_get[1,0]+1)/self.nn_get[1]])
        self.t_target = 0
        self.dt = dt
        self.pose_agent = pose_agent
        self.rng_xy = (self.pose_agent[0:2] + self.rng_xy0.T).T

        # map coordinate and vtgt
        self.vtgt_obj = VTgtSink(self.rng_xy, res_map=self.res_map,
                rng_get=self.rng_get, res_get=self.res_get) 

        # create first sink
        del_p_sink_r = np.random.uniform(self.rng_p_sink_r_th[0,0], self.rng_p_sink_r_th[0,1])
        del_p_sink_th = np.random.uniform(self.rng_p_sink_r_th[1,0], self.rng_p_sink_r_th[1,1])
        del_p_sink_x = np.cos(del_p_sink_th)*del_p_sink_r
        del_p_sink_y = np.sin(del_p_sink_th)*del_p_sink_r
        self.path_th = del_p_sink_th
        self.p_sink = self.pose_agent[0:2] + np.array([del_p_sink_x, del_p_sink_y])
        self.create_vtgt_sink(self.v_amp_rng)

# -----------------------------------------------------------------------------------------------------------------
    def create_vtgt_sink(self, v_amp_rng):
        d_sink = np.linalg.norm(self.p_sink - self.pose_agent[0:2])
        v_phase0 = np.random.uniform(-np.pi, np.pi)
        self.t_target0 = np.random.uniform(2, 4)
        self.vtgt_obj.create_vtgt_sink(self.p_sink, d_sink, v_amp_rng, v_phase0=v_phase0)

# -----------------------------------------------------------------------------------------------------------------
    def update(self, pose):
        self.pose_agent = pose

        if np.linalg.norm(self.p_sink - self.pose_agent[0:2]) < self.r_target:
            self.t_target += self.dt
        else: # reset t_target if agent goes out of 
            self.t_target = 0

        flag_new_target = 0
        if self.t_target > self.t_target0:
            del_p_sink_r = np.random.uniform(self.rng_p_sink_r_th[0,0], self.rng_p_sink_r_th[0,1])
            del_p_sink_th = np.random.uniform(self.rng_p_sink_r_th[1,0], self.rng_p_sink_r_th[1,1])
            self.path_th += del_p_sink_th
            del_p_sink_x = np.cos(self.path_th)*del_p_sink_r
            del_p_sink_y = np.sin(self.path_th)*del_p_sink_r
            self.p_sink += np.array([del_p_sink_x, del_p_sink_y])
            self.rng_xy = (self.pose_agent[0:2] + self.rng_xy0.T).T
            self.vtgt_obj.create_map(self.rng_xy)
            self.create_vtgt_sink(self.v_amp_rng)
            self.t_target = 0
            flag_new_target = 1

        return self.vtgt_obj.get_vtgt_field_local(pose), flag_new_target

# -----------------------------------------------------------------------------------------------------------------
    def get_vtgt(self, xy):
        return self.vtgt_obj.get_vtgt(xy)

# -----------------------------------------------------------------------------------------------------------------
    def get_vtgt_field_local(self, pose):
        return self.vtgt_obj.get_vtgt_field_local(pose)


class VTgtSink(object):
# -----------------------------------------------------------------------------------------------------------------
    def __init__(self, rng_xy=np.array([[-30, 30], [-30, 30]]), res_map=np.array([2, 2]),
                rng_get=np.array([[-5, 5], [-5, 5]]), res_get=np.array([2, 2]) ):
        # set parameters
        self.res_map = res_map
        self.res_get = res_get
        self.rng_get = rng_get

        # map coordinate and vtgt
        self.create_map(rng_xy)
        self.vtgt = -1*self.map

# -----------------------------------------------------------------------------------------------------------------
    def __del__(self):
        nn = "empty"

# -----------------------------------------------------------------------------------------------------------------
    def create_map(self, rng_xy):
        self.map_rng_xy = rng_xy
        self.map = self._generate_grid(rng_xy, self.res_map)

# -----------------------------------------------------------------------------------------------------------------
    def create_vtgt_sink(self, p_sink, d_sink, v_amp_rng, v_phase0=np.random.uniform(-np.pi, np.pi)):
        # set vtgt orientations
        rng_xy = (-p_sink + self.map_rng_xy.T).T
        self.vtgt = -self._generate_grid(rng_xy, self.res_map)

        # set vtgt amplitudes
        self._set_sink_vtgt_amp(p_sink, d_sink, v_amp_rng, v_phase0)

        self.vtgt_interp_x = interpolate.interp2d(self.map[0,:,0], self.map[1,0,:], self.vtgt[0].T, kind='linear')
        self.vtgt_interp_y = interpolate.interp2d(self.map[0,:,0], self.map[1,0,:], self.vtgt[1].T, kind='linear')

# -----------------------------------------------------------------------------------------------------------------
    def get_vtgt(self, xy):
        vtgt_x = self.vtgt_interp_x(xy[0], xy[1])
        vtgt_y = self.vtgt_interp_y(xy[0], xy[1])
        return np.array([vtgt_x, vtgt_y])

# -----------------------------------------------------------------------------------------------------------------
    def get_vtgt_field_local(self, pose):
        xy = pose[0:2]
        th = pose[2]

        # create query map
        get_map0 = self._generate_grid(self.rng_get, self.res_get)
        get_map_x = np.cos(th)*get_map0[0,:,:] - np.sin(th)*get_map0[1,:,:] + xy[0]
        get_map_y = np.sin(th)*get_map0[0,:,:] + np.cos(th)*get_map0[1,:,:] + xy[1]

        # get vtgt
        vtgt_x0 = np.reshape(np.array([self.vtgt_interp_x(x, y) \
                            for x, y in zip(get_map_x.flatten(), get_map_y.flatten())]),
                            get_map_x.shape)
        vtgt_y0 = np.reshape(np.array([self.vtgt_interp_y(x, y) \
                            for x, y in zip(get_map_x.flatten(), get_map_y.flatten())]),
                            get_map_y.shape)

        vtgt_x = np.cos(-th)*vtgt_x0 - np.sin(-th)*vtgt_y0
        vtgt_y = np.sin(-th)*vtgt_x0 + np.cos(-th)*vtgt_y0

        # debug
        """
        if xy[0] > 10:
            import matplotlib.pyplot as plt
            plt.figure(100)
            plt.axes([.025, .025, .95, .95])
            plt.plot(get_map_x, get_map_y, '.')
            plt.axis('equal')

            plt.figure(101)
            plt.axes([.025, .025, .95, .95])
            R = np.sqrt(vtgt_x0**2 + vtgt_y0**2)
            plt.quiver(get_map_x, get_map_y, vtgt_x0, vtgt_y0, R)
            plt.axis('equal')

            plt.show()
        """

        return np.stack((vtgt_x, vtgt_y))

# -----------------------------------------------------------------------------------------------------------------
    def _generate_grid(self, rng_xy=np.array([[-10, 10], [-10, 10]]), res=np.array([2, 2])):
        xo = .5*(rng_xy[0,0]+rng_xy[0,1])
        x_del = (rng_xy[0,1]-xo)*res[0]
        yo = .5*(rng_xy[1,0]+rng_xy[1,1])
        y_del = (rng_xy[1,1]-yo)*res[1]
        grid = np.mgrid[-x_del:x_del+1, -y_del:y_del+1]
        grid[0] = grid[0]/res[0] + xo
        grid[1] = grid[1]/res[1] + yo
        return grid

# -----------------------------------------------------------------------------------------------------------------
    def _set_sink_vtgt_amp(self, p_sink, d_sink, v_amp_rng, v_phase0, d_dec = 1):
        # d_dec: start to decelerate within d_dec of sink

        for i_x, x in enumerate(self.map[0,:,0]):
            for i_y, y in enumerate(self.map[1,0,:]):
                d = np.linalg.norm([ x-p_sink[0], y-p_sink[1] ])
                if d > d_sink + d_dec:
                    v_amp = v_amp_rng[1]
                elif d > d_dec:
                    v_phase = v_phase0 + d/d_sink*2*np.pi
                    v_amp = .5*np.diff(v_amp_rng)*np.sin(v_phase) + np.mean(v_amp_rng)
                else:
                    v_phase = v_phase0 + d_dec/d_sink*2*np.pi
                    v_amp0 = .5*np.diff(v_amp_rng)*np.sin(v_phase) + np.mean(v_amp_rng)
                    v_amp = d*v_amp0

                amp_norm = np.linalg.norm(self.vtgt[:,i_x,i_y])
                self.vtgt[0,i_x,i_y] = v_amp*self.vtgt[0,i_x,i_y]/amp_norm
                self.vtgt[1,i_x,i_y] = v_amp*self.vtgt[1,i_x,i_y]/amp_norm