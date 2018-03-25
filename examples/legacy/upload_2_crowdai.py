#!/usr/bin/env python
try:
    import crowdai
except:
    raise Exception("Please install the `crowdai` python client by : pip install crowdai")
import argparse
parser = argparse.ArgumentParser(description='Upload saved docker environments to crowdai for grading')
parser.add_argument('--api_key', dest='api_key', action='store', required=True)
parser.add_argument('--docker_container', dest='docker_container', action='store', required=True)
args = parser.parse_args()

challenge = crowdai.Challenge("Learning2RunChallengeNIPS2017", args.api_key)
result = challenge.submit(args.docker_container)

print(result)
