from gym.envs.registration import register


env_id = 'fourroom-v0'

register(
        id=env_id,
        entry_point='four_rooms:FourRooms',
        kwargs={
            'goal': (10, 10),
            'viz_params': ['pixel'],
        },
        max_episode_steps=500
        )