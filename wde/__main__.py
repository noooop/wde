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
    from wde.microservices.framework.zero_manager.client import \
        ZeroManagerClient

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
    from wde.microservices.framework.zero_manager.client import \
        ZeroManagerClient

    manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
    out = manager_client.terminate(model_name)
    logger.info("%s", out)


@click.group()
def main():
    pass


main.add_command(server)
main.add_command(start)
main.add_command(terminate)

if __name__ == '__main__':
    main()
