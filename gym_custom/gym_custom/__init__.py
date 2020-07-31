from gym.envs.registration import register

register(id='ImPendulum-v0',
        entry_point='gym_custom.envs:ImPendulumEnv',
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

register(id='PendulumPos-v0',
        entry_point='gym_custom.envs:PendulumPosEnv',
        max_episode_steps=60,
)

register(id='AcrobotPos-v0',
        entry_point='gym_custom.envs:AcrobotPosEnv',
)

register(id='QAcrobot-v0',
        entry_point='gym_custom.envs:QAcrobotEnv',
        max_episode_steps=50,
)

register(id='QAcrobotPos-v0',
        entry_point='gym_custom.envs:QAcrobotPosEnv',
)

register(id='SpringMassPos-v0',
        entry_point='gym_custom.envs:SpringMassPosEnv',
)

register(id='CartpoleMod-v0',
        entry_point='gym_custom.envs:CartPoleEnv',
        max_episode_steps=50,
)

register(id='Quadrotor2D-v0',
        entry_point='gym_custom.envs:Quadrotor2DEnv',
        max_episode_steps=100,
)
register(id='CoupledMassSpring-v0',
        entry_point='gym_custom.envs:CoupledMassSpringEnv',
        max_episode_steps=50,
)
register(id='Manipulator-v0',
        entry_point='gym_custom.envs:ManipulatorEnv',
        max_episode_steps=50,
)
