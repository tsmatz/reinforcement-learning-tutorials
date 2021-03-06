# Reinforcement Learning Tutorials

This content shows you what's doing in typical reinforcement learning (RL) methods with several lines of code.

1. [Q-Learning](01-q-learning.ipynb)
2. [Policy Gradient](02-policy-gradient.ipynb)
3. [Actor Critic](03-actor-critic.ipynb)
4. [Deep Reinforcement Learning in Minecraft](https://github.com/tsmatz/malmo-maze-sample)

Through these contents in this repository, CartPole environemnt is used for running RL.<br>
See below about specs (action space, observation space, and rewards) for this environment.

<u>sample code</u>

```
import gym
import random

def pick_sample():
  return random.randint(0, 1)

env = gym.make("CartPole-v0")
for i in range(1):
  print("start episode {}".format(i))
  done = False
  s = env.reset()
  while not done:
    a = pick_sample()
    s, r, done, _ = env.step(a)
    print("action: {},  reward: {}".format(a, r))
    print("state: {}, {}, {}, {}".format(s[0], s[1], s[2], s[3]))
env.close()
```

<u>output</u>

```
start episode 0
action: 0,  reward: 1.0
state: 0.006784938861824417, -0.18766506871206354, 0.0287443864274386, 0.27414982492533896
action: 0,  reward: 1.0
state: 0.0030316374875831464, -0.383185104857609, 0.03422738292594538, 0.5757584135859465
action: 1,  reward: 1.0
state: -0.004632064609569034, -0.18855925062821827, 0.04574255119766431, 0.2940515065957076
```

Action Space - Discrete(2) :<br>
- 0 : Push cart to the left
- 1 : Push cart to the right

Observation Space - Box(-num, num, (4,), float32) :<br>
- Cart Position (-4.8, 4.8)
- Cart Velocity (-inf, inf)
- Pole Angle (-0.41, 0.41)
- Pole Velocity At Tip (-inf, inf)

Reward - float32 :<br>
It always returns 1.0 as reward. If succeeded, you can take max 200 rewards in a single episode. (This will be the goal for learning.)

> Note : Call ```render()``` when you want to show current state in visual UI.<br>
> ![CartPole rendering](assets/cart-pole.png?raw=true)

*Tsuyoshi Matsuzaki @ Microsoft*
