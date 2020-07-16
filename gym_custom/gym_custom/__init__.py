from gym.envs.registration import register

register(id='QPendulum-v0',
        entry_point='gym_custom.envs:QPendulumEnv',
        max_episode_steps=60,
)

register(id='SpringMass-v0',
        entry_point='gym_custom.envs:SpringMassEnv',
)

register(id='PendulumPos-v0',
        entry_point='gym_custom.envs:PendulumPosEnv',
)

register(id='AcrobotPos-v0',
        entry_point='gym_custom.envs:AcrobotPosEnv',
)

register(id='QAcrobot-v0',
        entry_point='gym_custom.envs:QAcrobotEnv',
        max_episode_steps=200,
)

register(id='QAcrobotPos-v0',
        entry_point='gym_custom.envs:QAcrobotPosEnv',
)

register(id='SpringMassPos-v0',
        entry_point='gym_custom.envs:SpringMassPosEnv',
)

register(id='CartpoleMod-v0',
        entry_point='gym_custom.envs:CartPoleEnv',
        max_episode_steps=200,
)
