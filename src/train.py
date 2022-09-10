import multiprocessing
import reverb

import jax

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
    multiprocessing.set_start_method("spawn")
    config = Config()
    builder = Builder(config)
    client = reverb.Client(f'localhost:{config.port}')
    server = multiprocessing.Process(target=make_server, args=(builder,))
    server.start()
    actors = []
    for i in range(config.num_actors):
        actor = multiprocessing.Process(target=make_actor, args=(builder,))
        actor.start()
        actors.append(actor)
    learner = multiprocessing.Process(target=make_learner, args=(builder,))
    learner.start()
    server.join()
    for actor in actors:
        actor.join()
    learner.join()

