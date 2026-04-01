gymnasium-http-api
==================

REST interface for the [Gymnasium](https://gymnasium.farama.org/) library to allow for language-agnostic development
of reinforcement learning agents. Spiritual successor to [openai/gym-http-api](https://github.com/openai/gym-http-api).

### Quick Start

The fastest way to get started is to pull and run the latest version of the associated container image from
[GHCR](https://ghcr.io/cloudkj/gymnasium-http-api):

```
docker run --rm -it -p 5000:5000 ghcr.io/cloudkj/gymnasium-http-api
```

Alternatively, you can clone the code and run the server directly:

```
python3 -m uvicorn app.main:app --port 5000 --host 0.0.0.0
```

Once the server is running, create and interact with environments through the various endpoints. Here's a  command-line example of creating an environment, resetting its state, then taking one action.

```
% curl -X POST http://localhost:5000/v1/envs/ -H 'Content-Type: application/json' -d '{"env_id":"CartPole-v1"}' 
{"instance_id":"986a40f7-6472-4ac6-bc5f-86b7939898b1"}

% curl -X POST http://localhost:5000/v1/envs/986a40f7-6472-4ac6-bc5f-86b7939898b1/reset/
{"observation":[-0.001753740361891687,0.0477466844022274,0.003655971959233284,-0.030443252995610237],"info":{}}

% curl -X POST http://localhost:5000/v1/envs/986a40f7-6472-4ac6-bc5f-86b7939898b1/step/ -H 'Content-Type: application/json' -d '{"action": 0}'
{"observation":[-0.0037473568227142096,-0.3425928056240082,0.008314925245940685,0.557033360004425],"reward":1.0,"terminated":false,"truncated":false,"info":{}}
```

### Demo

To see an agent in action, check out the standalone, client-side Javascript [agent](https://github.com/cloudkj/gymnasium-http-api/blob/main/app/static/agent.html) available at `/agent.html` which shows a typical agent loop over a single episode. You can also modify the agent code directly in your browser and try out different heuristics or policies.

To observe the state of all active environments on the server, check out the [monitoring](https://github.com/cloudkj/gymnasium-http-api/blob/main/app/static/monitor.html) page available at `/monitor.html`.

### Usage

To start developing an agent, simply call endpoints to create and interact with the environment of your
choice. Here's a simple Python example that wraps the main endpoints as a drop-in replacement for
[`gymnasium.Env`](https://gymnasium.farama.org/api/env/):

```python
import requests

class Env:
    BASE_URL = "http://localhost:5000/v1/envs"
    def __init__(self, env_id): self.instance_id = requests.post(f"{self.BASE_URL}/", json={"env_id": env_id}).json()["instance_id"]
    def reset(self): return requests.post(f"{self.BASE_URL}/{ self.instance_id}/reset/", json={}).json()
    def step(self, action): return requests.post(f"{self.BASE_URL}/{self.instance_id}/step/", json={"action": action}).json()
    def close(self): requests.delete(f"{self.BASE_URL}/{self.instance_id}/")

# Create an environment and reset state to start a new episode
env = Env("CartPole-v1")
initial = env.reset()
observation, info = initial["observation"], initial["info"]
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    action = 0 # To the left, to the left
    # Take the action and see what happens
    state = env.step(action)
    total_reward += state["reward"]
    episode_over = state["terminated"] or state["truncated"]

print(f"Episode finished! Total reward: {total_reward}")
env.close()
```

### Documentation

The endpoints largely follow the conventions established by the legacy [gym-http-api](https://github.com/openai/gym-http-api)
project. For the latest documentation, start an instance of the server and navigate to `/docs` to
view auto-generated documentation for all supported endpoints.

##### Environments

* `GET /v1/envs/` - List all active environment instances
* `POST /v1/envs/` - Create an instance of the specified environment
* `POST /v1/envs/{instance_id}/reset/` - Reset the environment
* `POST /v1/envs/{instance_id}/step/` - Step through the environment with a specified action
* `GET /v1/envs/{instance_id}/action_space/` - Get action space properties
* `GET /v1/envs/{instance_id}/observation_space/` - Get observation space properties
* `DELETE /v1/envs/{instance_id}/` - Close and remove the environment

##### Monitoring

* `GET /v1/envs/{instance_id}/monitor/render/` - Render the current state of the environment
* `GET /v1/envs/{instance_id}/monitor/stream/` - Continuously stream the current state of the environment
