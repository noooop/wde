from wde import const, envs
from wde.microservices.framework.zero.server import ZeroServerProcess


class Server:

    def __init__(self):
        self.manager = None
        self.nameserver = None

    def setup(self):
        self.nameserver = ZeroServerProcess(const.NAMESERVER_CLASS)
        self.manager = ZeroServerProcess(const.MANAGER_CLASS,
                                         server_kwargs={
                                             "name": envs.ROOT_MANAGER_NAME,
                                             "server_class":
                                             const.MANAGER_CLASS
                                         })

    def run(self, waiting=True):
        self.nameserver.start()
        self.nameserver.wait_port_available()

        if self.nameserver.status != "running":
            return

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


if __name__ == '__main__':
    server = Server()
    server.setup()
    server.run()
