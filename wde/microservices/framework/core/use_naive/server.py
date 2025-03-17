import signal
import traceback
from multiprocessing import Event, Pipe, Process, Value

import zmq
from zmq.error import ZMQError

from wde.logger import init_logger
from wde.utils import lazy_import

logger = init_logger(__name__)


class ZeroServerNaiveImpl:
    POLL_INTERVAL = 1000

    def __init__(self,
                 engine_class: str,
                 engine_kwargs=None,
                 port=None,
                 event=None,
                 share_port=None,
                 *args,
                 **kwargs):
        context = zmq.Context.instance()
        socket = context.socket(zmq.ROUTER)

        if port is None or port == "random":
            port = socket.bind_to_random_port("tcp://*",
                                              min_port=50000,
                                              max_port=60000)
        else:
            try:
                socket.bind(f"tcp://*:{port}")
            except ZMQError:
                share_port.value = -2
                return

        if share_port is not None:
            share_port.value = int(port)

        self.socket = socket
        self.port = port

        self.event = event if event is not None else Event()
        self.event.set()
        self.engine_class = engine_class
        self.engine_kwargs = engine_kwargs or dict()
        self.engine = None

    def run(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while self.event.is_set():
            try:
                socks = dict(poller.poll(self.POLL_INTERVAL))
            except (KeyboardInterrupt, EOFError):
                return

            if socks.get(self.socket) == zmq.POLLIN:
                msg = self.socket.recv_multipart(copy=False)
                self.engine(msg)

    def init(self):
        self.engine = lazy_import(self.engine_class)(socket=self.socket,
                                                     port=self.port,
                                                     **self.engine_kwargs)


class ZeroServerProcess(Process):
    _status_code = [
        "prepare", "started", "initial", "running", "error", "stopped"
    ]
    status_code2str = {c: s for c, s in enumerate(_status_code)}
    status_str2code = {s: c for c, s in enumerate(_status_code)}

    def __init__(self,
                 engine_class,
                 engine_kwargs=None,
                 server_kwargs=None,
                 event=None,
                 ignore_warnings=False,
                 debug=False):
        Process.__init__(self, )

        if event is None:
            self.event = Event()
        else:
            self.event = event

        self.debug = debug
        self.engine_class = engine_class
        self.engine_kwargs = engine_kwargs or dict()
        self.server_kwargs = server_kwargs or dict()
        self.server = None
        self.share_port = Value('i', -1)
        self.ignore_warnings = ignore_warnings

        self._status = Value('i', 0)
        self._parent_conn, self._child_conn = Pipe()
        self._exception = None

        self._set_status("prepare")

    def _set_status(self, status):
        self._status.value = self.status_str2code[status]

    @property
    def status(self):
        return self.status_code2str[self._status.value]

    @property
    def exception(self):
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception

    def run(self):
        self._set_status("started")

        try:
            if self.ignore_warnings:
                import warnings
                warnings.filterwarnings("ignore")

            self.server_kwargs["event"] = self.event
            self.server_kwargs["share_port"] = self.share_port

            self.server = ZeroServerNaiveImpl(engine_class=self.engine_class,
                                              engine_kwargs=self.engine_kwargs,
                                              **self.server_kwargs)
        except (FileNotFoundError, EnvironmentError) as e:
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return
        except Exception as e:
            traceback.print_exc()
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

        self._set_status("initial")

        try:
            self.server.init()
        except (FileNotFoundError, EnvironmentError) as e:
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return
        except Exception as e:
            traceback.print_exc()
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

        self._set_status("running")

        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            self.server.run()
        except Exception as e:
            traceback.print_exc()
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

    def signal_handler(self, signum, frame):
        self.close()

    def close(self):
        self.event.clear()
        self.join()
        self._set_status("stopped")

    def terminate(self):
        self.close()

    def wait(self):
        try:
            self.join()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self.terminate()
