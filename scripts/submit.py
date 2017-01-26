from osim.http.client import Client

if __name__ == '__main__':
    CROWDAI_TOKEN = "TOKEN_TEST"
    remote_base = 'http://54.154.84.135:80'
    client = Client(remote_base)

    # Create environment
    env_id = "Gait" #'CartPole-v0'
    instance_id = client.env_create(env_id, CROWDAI_TOKEN)

    # Run a single step
    client.env_monitor_start(instance_id, directory='tmp', force=True)
    init_obs = client.env_reset(instance_id)
    for i in range(500):
        [observation, reward, done, info] = client.env_step(instance_id, [0.0]*18, True)
        if done:
            break
    client.env_monitor_close(instance_id)
    client.env_close(instance_id)

