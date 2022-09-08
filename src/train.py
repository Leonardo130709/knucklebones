import multiprocessing
import reverb

import jax

from src.alg.builder import Builder
from src.alg.config import Config


def make_server():
    config = Config()
    builder = Builder(config)
    tables = builder.make_tables()
    server = reverb.Server(tables, config.port)
    # return server
    server.wait()


def make_actor():
    config = Config()
    builder = Builder(config)
    client = reverb.Client(f'localhost:{config.port}')
    rng = jax.random.PRNGKey(config.seed)
    actor = builder.make_actor(rng, client)
    actor.run()


def make_learner():
    config = Config()
    builder = Builder(config)
    client = reverb.Client(f'localhost:{config.port}')
    rng = jax.random.PRNGKey(config.seed)
    learner = builder.make_learner(rng, client)
    learner.run()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    # server = make_server()
    # client = server.localhost_client()
    # config = Config()
    # builder = Builder(config)
    # key = jax.random.PRNGKey(0)
    # learner = builder.make_learner(key, client)
    # actor = builder.make_actor(key, client)
    #
    # actor.run()
    # learner.run()


    server = multiprocessing.Process(target=make_server)
    actor = multiprocessing.Process(target=make_actor)
    learner = multiprocessing.Process(target=make_learner)
    server.start()
    actor.start()
    learner.start()
    server.join()
    actor.join()
    learner.join()

