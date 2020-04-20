from gym.envs.registration import register

register(id='QPendulum-v0',
        entry_point='gym_custom.envs:QPendulumEnv',
)
