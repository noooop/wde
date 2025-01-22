from wde import const, envs
from wde.microservices.framework.zero.server import ZeroServerProcess


class Server:

    def __init__(self, bind_random_port=False):
        self.manager = None
        self.nameserver = None
        self.bind_random_port = bind_random_port
        self.nameserver_port = None

    def setup(self):
        if self.bind_random_port:
            server_kwargs = {"port": "random"}
        else:
            server_kwargs = {}

        self.nameserver = ZeroServerProcess(const.NAMESERVER_CLASS,
                                            server_kwargs)

    def run(self, waiting=True):
        self.nameserver.start()
        self.nameserver_port = self.nameserver.wait_port_available()

        if self.nameserver.status != "running":
            return

        self.manager = ZeroServerProcess(const.MANAGER_CLASS,
                                         server_kwargs={
                                             "name":
                                             envs.ROOT_MANAGER_NAME,
                                             "server_class":
                                             const.MANAGER_CLASS,
                                             "nameserver_port":
                                             self.nameserver_port
                                         })

        self.manager.start()

        if waiting:
            self.wait()

    def wait(self):
        for h in [self.nameserver, self.manager]:
            if h is None:
                continue
            h.wait()

    def terminate(self):
        for h in [self.nameserver, self.manager]:
            if h is None:
                continue
            h.terminate()


def setup_and_run():
    server = Server()
    server.setup()
    server.run(waiting=False)
    return server


if __name__ == '__main__':
    server = Server()
    server.setup()
    server.run()
