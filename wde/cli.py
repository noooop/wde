import click

from wde import envs
from wde.logger import init_logger

logger = init_logger(__name__)


@click.command()
def server():
    from wde.microservices.standalone.server import Server

    server = Server()
    server.setup()
    server.run()


@click.command()
@click.argument('model_name')
@click.option("--wait/--nowait", default=True)
def start(model_name, wait):
    from wde.client import ZeroManagerClient

    manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
    out = manager_client.start(model_name)

    logger.info("%s", out)

    if not wait:
        return

    logger.info("Wait %s available.", model_name)
    manager_client.wait_service_status(model_name)


@click.command()
@click.argument('model_name')
def terminate(model_name):
    from wde.client import ZeroManagerClient

    manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
    out = manager_client.terminate(model_name)
    logger.info("%s", out)


@click.command()
@click.argument('config_filename', type=click.Path(exists=True))
def deploy(config_filename):
    from wde.microservices.standalone.deploy import Deploy
    Deploy(config_filename)()


@click.command()
@click.argument('config_filename', type=click.Path(exists=True))
def serving(config_filename):
    from wde.client import ZeroManagerClient
    from wde.microservices.standalone.deploy import Deploy
    from wde.microservices.standalone.server import Server

    server = Server()
    server.setup()
    server.run(waiting=False)

    manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
    manager_client.wait_service_available(envs.ROOT_MANAGER_NAME)

    Deploy(config_filename)()

    server.wait()


@click.group()
def main():
    pass


main.add_command(server)
main.add_command(deploy)
main.add_command(serving)
main.add_command(start)
main.add_command(terminate)

if __name__ == '__main__':
    main()
