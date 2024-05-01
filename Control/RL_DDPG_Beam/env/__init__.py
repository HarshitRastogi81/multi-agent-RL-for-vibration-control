from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)


register(
    id='BeamEnvironment-v0',
    entry_point='env.beam_env.BeamEnvironment:BeamEnvironment',
)
load_plugin_envs()