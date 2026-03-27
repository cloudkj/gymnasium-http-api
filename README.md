gymnasium-http-api
==================

RESTful interface for the [Gymnasium](https://gymnasium.farama.org/) library to allow for language-agnostic development
of reinforcement learning agents. Spiritual successor to [gym-http-api](https://github.com/openai/gym-http-api).

## Quick Start

The fastest way to get started is to pull and run the latest version of the associated container image from
[GHCR](https://ghcr.io/cloudkj/gymnasium-http-api):

```
docker run --rm -it -p 5000:5000 ghcr.io/cloudkj/gymnasium-http-api
```

Once the server is running, you can create and interact with environments through various endpoints. For example:

```
curl -X POST -H "Content-Type: application/json" -d '{"env_id":"CartPole-v1"}' http://localhost:5000/v1/envs/
```

Navigate to http://localhost:5000/docs to view auto-generated documentation for all supported endpoints.

### Demo

To see an agent in action, check out the standalone, client-side Javascript agent available at
http://localhost:5000/agent.html ([source](https://github.com/cloudkj/gymnasium-http-api/blob/main/app/static/agent.html))
which shows a typical agent loop over a single episode. You can also modify the agent code directly in your browser and
try out different heuristics or policies.

### Basic Usage

To start developing an agent, simply call endpoints to create and interact with the environment of your
choice. Here's a simple example that wraps the main endpoints as a drop-in replacement for
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
