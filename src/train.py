import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import warnings
warnings.filterwarnings('ignore')


import multiprocessing as mp
import reverb

from src.alg.builder import Builder
from src.alg.config import Config


def make_server(builder):
    server = builder.make_server()
    server.wait()


def make_actor(builder):
    actor = builder.make_actor()
    actor.run()


def make_learner(builder):
    learner = builder.make_learner()
    learner.run()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    config = Config()
    builder = Builder(config)
    client = reverb.Client(f'localhost:{config.port}')
    server = mp.Process(target=make_server, args=(builder,))
    learner = mp.Process(target=make_learner, args=(builder,))
    server.start()
    learner.start()
    actors = []
    num_actors = min(config.num_actors, mp.cpu_count())
    for i in range(num_actors):
        actor = mp.Process(target=make_actor, args=(builder,))
        actor.start()
        actors.append(actor)
    for actor in actors:
        actor.join()
    learner.join()
    server.join()

