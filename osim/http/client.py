# ADAPTED FROM https://github.com/openai/gym-http-api
import requests
import six.moves.urllib.parse as urlparse
import json
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Client(object):
    """
    Gym client to interface with gym_http_server
    """
    def __init__(self, remote_base):
        self.remote_base = remote_base
        self.session = requests.Session()
        self.session.headers.update({'Content-type': 'application/json'})
        self.instance_id = None

    def _parse_server_error_or_raise_for_status(self, resp):
        j = {}
        try:
            j = resp.json()
        except:
            # Most likely json parse failed because of network error, not server error (server
            # sends its errors in json). Don't let parse exception go up, but rather raise default
            # error.
            resp.raise_for_status()
        if resp.status_code != 200 and "message" in j:  # descriptive message from server side
            raise ServerError(message=j["message"], status_code=resp.status_code)
        resp.raise_for_status()
        return j

    def _post_request(self, route, data):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("POST {}\n{}".format(url, json.dumps(data)))
        resp = self.session.post(urlparse.urljoin(self.remote_base, route),
                            data=json.dumps(data))
        return self._parse_server_error_or_raise_for_status(resp)

    def _get_request(self, route):
        url = urlparse.urljoin(self.remote_base, route)
        logger.info("GET {}".format(url))
        resp = self.session.get(url)
        return self._parse_server_error_or_raise_for_status(resp)

    def env_create(self, token, env_id = "Run"):
        route = '/v1/envs/'
        data = {'env_id': env_id,
                'token': token}
        resp = self._post_request(route, data)
        self.instance_id = resp['instance_id']
        self.env_monitor_start("tmp", force=True)
        return self.env_reset()

    def env_reset(self):
        route = '/v1/envs/{}/reset/'.format(self.instance_id)
        resp = self._post_request(route, None)
        observation = resp['observation']
        return observation

    def env_step(self, action, render=False):
        route = '/v1/envs/{}/step/'.format(self.instance_id)
        data = {'action': action, 'render': render}
        resp = self._post_request(route, data)
        observation = resp['observation']
        reward = resp['reward']
        done = resp['done']
        info = resp['info']
        return [observation, reward, done, info]

    def env_monitor_start(self, directory,
                              force=False, resume=False, video_callable=False):
        route = '/v1/envs/{}/monitor/start/'.format(self.instance_id)
        data = {'directory': directory,
                'force': force,
                'resume': resume,
                'video_callable': video_callable}
        self._post_request(route, data)

    def submit(self):
        route = '/v1/envs/{}/monitor/close/'.format(self.instance_id)
        result = self._post_request(route, None)
        if result['reward']:
            print("Your total reward from this submission: %f" % result['reward'])
        else:
            print("There was an error in your submission. Please contact administrators.")
        route = '/v1/envs/{}/close/'.format(self.instance_id)
        self.env_close()

    def env_close(self):
        route = '/v1/envs/{}/close/'.format(self.instance_id)
        self._post_request(route, None)

class ServerError(Exception):
    def __init__(self, message, status_code=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
