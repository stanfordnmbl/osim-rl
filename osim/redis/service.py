#!/usr/bin/env python
from __future__ import print_function
import redis
from osim.redis import messages
import json
import numpy as np
import osim
from osim.env import *
import os
import timeout_decorator
import time



class OsimRlRedisService:
    def __init__(self, osim_rl_redis_service_id='osim_rl_redis_service_id', seed_map=False, max_steps=1000, remote_host='127.0.0.1', remote_port=6379, remote_db=0, remote_password=None, visualize=False, verbose=False):
        """
            TODO: Expose more RunEnv related variables
        """
        print("Attempting to connect to redis server at {}:{}/{}".format(remote_host, remote_port, remote_db))
        self.redis_pool = redis.ConnectionPool(host=remote_host, port=remote_port, db=remote_db, password=remote_password)
        self.namespace = "osim-rl"
        self.service_id = osim_rl_redis_service_id
        self.command_channel = "{}::{}::commands".format(self.namespace, self.service_id)
        self.env = False
        self.env_available = False
        self.reward = 0
        self.simulation_count = 0
        self.simualation_rewards = []
        self.simulation_times = []
        self.begin_simulation = False
        self.current_step = 0
        self.verbose = verbose
        self.visualize = visualize
        self.max_steps = max_steps
        self.initalize_seed_map(seed_map)

    def initalize_seed_map(self, seed_map_string):
        if seed_map_string:
            assert type(seed_map_string) == type("")
            seed_map = seed_map_string.split(",")
            seed_map = [int(x) for x in seed_map]
            self.seed_map = seed_map
        else:
            self.seed_map = [np.random.randint(0,10**10)]

    def get_redis_connection(self):
        return redis.Redis(connection_pool=self.redis_pool)

    def _error_template(self, payload):
        _response = {}
        _response['type'] = messages.OSIM_RL.ERROR
        _response['payload'] = payload
        return _response

    @timeout_decorator.timeout(15*60)#15*60 seconds timeout for each command
    def get_next_command(self, _redis):
        command = _redis.brpop(self.command_channel)[1]
        return command

    def run(self):
        print("Listening for commands at : ", self.command_channel)
        while True:
            try:
                _redis = self.get_redis_connection()
                command = self.get_next_command(_redis)
            except timeout_decorator.timeout_decorator.TimeoutError:
                raise Exception("Timeout in step {} of simulation {}".format(self.current_step, self.simulation_count))
            command_response_channel = "default_response_channel"
            if self.verbose: print("Self.Reward : ", self.reward)
            if self.verbose: print("Current Simulation : ", self.simulation_count)
            if self.seed_map and self.verbose and self.simulation_count < len(self.seed_map): print("Current SEED : ", self.seed_map[self.simulation_count])
            try:
                command = json.loads(command.decode('utf-8'))
                if self.verbose: print("Received Request : ", command)
                command_response_channel = command['response_channel']
                if command['type'] == messages.OSIM_RL.PING:
                    """
                        INITIAL HANDSHAKE : Respond with PONG
                    """
                    _command_response = {}
                    _command_response['type'] = messages.OSIM_RL.PONG
                    _command_response['payload'] = {}
                    if self.verbose: print("Responding with : ", _command_response)
                    _redis.rpush(command_response_channel, json.dumps(_command_response))
                elif command['type'] == messages.OSIM_RL.ENV_CREATE:
                    """
                        ENV_CREATE

                        Respond with initial observation
                    """
                    _payload = command['payload']

                    if self.env: #If env already exists, throw an error
                        _error_message = "Attempt to create environment when one already exists."
                        if self.verbose: print("Responding with : ", self._error_template(_error_message))
                        _redis.rpush( command_response_channel, self._error_template(_error_message))
                        return self._error_template(_error_message)
                    else:
                        self.env = RunEnv(visualize = self.visualize, max_obstacles=10)
                        _observation = self.env.reset(seed=self.seed_map[self.simulation_count], difficulty=2)
                        self.begin_simulation = time.time()
                        self.simualation_rewards.append(0)
                        self.env_available = True
                        self.current_step = 0
                        _observation = np.array(_observation).tolist()
                        _command_response = {}
                        _command_response['type'] = messages.OSIM_RL.ENV_CREATE_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = _observation
                        if self.verbose: print("Responding with : ", _command_response)
                        _redis.rpush(command_response_channel, json.dumps(_command_response))
                elif command['type'] == messages.OSIM_RL.ENV_RESET:
                    """
                        ENV_RESET

                        Respond with observation from next simulation or
                        False if no simulations are left
                    """
                    self.simulation_count += 1
                    if self.begin_simulation:
                        self.simulation_times.append(time.time()-self.begin_simulation)
                        self.begin_simulation = time.time()
                    if self.seed_map and self.simulation_count < len(self.seed_map):
                        _observation = self.env.reset(seed=self.seed_map[self.simulation_count], difficulty=2)
                        self.simualation_rewards.append(0)
                        self.env_available = True
                        self.current_step = 0
                        _observation = list(_observation)
                        _command_response = {}
                        _command_response['type'] = messages.OSIM_RL.ENV_RESET_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = _observation
                        if self.verbose: print("Responding with : ", _command_response)
                        _redis.rpush(command_response_channel, json.dumps(_command_response))
                    else:
                        _command_response = {}
                        _command_response['type'] = messages.OSIM_RL.ENV_RESET_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = False
                        if self.verbose: print("Responding with : ", _command_response)
                        _redis.rpush(command_response_channel, json.dumps(_command_response))
                elif command['type'] == messages.OSIM_RL.ENV_STEP:
                    """
                        ENV_STEP

                        Request : Action array
                        Respond with updated [observation,reward,done,info] after step
                    """
                    args = command['payload']
                    action = args['action']
                    action = np.array(action)
                    if self.env and self.env_available:
                        [observation, reward, done, info] = self.env.step(action)
                    else:
                        if self.env:
                            raise Exception("Attempt to call `step` function after max_steps={} in a single simulation. Please reset your environment before calling the `step` function after max_step s".format(self.max_steps))
                        else:
                                raise Exception("Attempt to call `step` function on a non existent `env`")
                    self.reward += reward
                    self.simualation_rewards[-1] += reward
                    self.current_step += 1
                    if self.current_step >= self.max_steps:
                        _command_response = {}
                        _command_response['type'] = messages.OSIM_RL.ENV_STEP_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = np.array(observation).tolist()
                        _command_response['payload']['reward'] = reward
                        _command_response['payload']['done'] = True
                        _command_response['payload']['info'] = info

                        """
                        Mark env as unavailable until next reset
                        """
                        self.env_available = False
                    else:
                        _command_response = {}
                        _command_response['type'] = messages.OSIM_RL.ENV_STEP_RESPONSE
                        _command_response['payload'] = {}
                        _command_response['payload']['observation'] = np.array(observation).tolist()
                        _command_response['payload']['reward'] = reward
                        _command_response['payload']['done'] = done
                        _command_response['payload']['info'] = info

                        if done:
                            """
                                Mark env as unavailable until next reset
                            """
                            self.env_available = False
                    if self.verbose: print("Responding with : ", _command_response)
                    if self.verbose: print("Current Step : ", self.current_step)
                    _redis.rpush(command_response_channel, json.dumps(_command_response))
                elif command['type'] == messages.OSIM_RL.ENV_SUBMIT:
                    """
                        ENV_SUBMIT

                        Submit the final cumulative reward
                    """
                    _response = {}
                    _response['type'] = messages.OSIM_RL.ENV_SUBMIT_RESPONSE
                    _payload = {}
                    _payload['mean_reward'] = np.float(self.reward)/len(self.seed_map) #Mean reward
                    _payload['simulation_rewards'] = self.simualation_rewards
                    _payload['simulation_times'] = self.simulation_times
                    _response['payload'] = _payload
                    _redis.rpush(command_response_channel, json.dumps(_response))
                    if self.verbose: print("Responding with : ", _response)
                    return _response
                else:
                    _error = self._error_template(
                                    "UNKNOWN_REQUEST:{}".format(
                                        json.dumps(command)))
                    if self.verbose: print("Responding with : ", json.dumps(_error))
                    _redis.rpush(command_response_channel, json.dumps(_error))
                    return _error

            except Exception as e:
                print("Error : ", str(e))
                _redis.rpush(   command_response_channel,
                                json.dumps(self._error_template(str(e))))
                return self._error_template(str(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
    parser.add_argument('--port', dest='port', action='store', required=True)
    args = parser.parse_args()

    grader = OsimRlRedisService(remote_port=int(args.port), seed_map="11,22,33", max_steps=1000, verbose=True)
    result = grader.run()
    if result['type'] == messages.OSIM_RL.ENV_SUBMIT_RESPONSE:
        cumulative_results = result['payload']
        print("Results : ", cumulative_results)
    elif result['type'] == messages.OSIM_RL.ERROR:
        error = result['payload']
        raise Exception("Evaluation Failed : {}".format(str(error)))
    else:
        #Evaluation failed
        print("Evaluation Failed : ", result['payload'])
