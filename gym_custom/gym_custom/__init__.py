from gym.envs.registration import register

register(id='ImPendulum-v0',
        entry_point='gym_custom.envs:ImPendulumEnv',
        max_episode_steps=50,
)

register(id='PendulumMod-v0',
        entry_point='gym_custom.envs:PendulumModEnv',
        max_episode_steps=50,
)

register(id='PendulumModPos-v0',
        entry_point='gym_custom.envs:PendulumModPosEnv',
        max_episode_steps=50,
)

register(id='QPendulum-v0',
        entry_point='gym_custom.envs:QPendulumEnv',
        max_episode_steps=50,
)

register(id='SpringMass-v0',
        entry_point='gym_custom.envs:SpringMassEnv',
        max_episode_steps=50,
)


register(id='SpringMassPos-v0',
        entry_point='gym_custom.envs:SpringMassPosEnv',
)

register(id='CartpoleMod-v0',
        entry_point='gym_custom.envs:CartPoleEnv',
        max_episode_steps=50,
)

register(id='CartpoleModPos-v0',
        entry_point='gym_custom.envs:CartPolePosEnv',
        max_episode_steps=50,
)

register(id='CoupledMassSpring-v0',
        entry_point='gym_custom.envs:CoupledMassSpringEnv',
        max_episode_steps=50,
)

