import abc, time, os
from queue import Empty, Full
import multiprocessing as mp


class BaseService(metaclass=abc.ABCMeta):
    on_queue = None

    def __init__(self) -> None:
        self.job_queue = mp.Queue(maxsize=100)
        self.result_queue = mp.Queue(maxsize=100)
        self.cmd_queue = mp.Queue(maxsize=4096)
        self.proc = mp.Process(target=self.proc_main, daemon=True)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        if 'proc' in state:
            del state["proc"]
        return state

    def start(self):
        self.proc.start()

    def proc_main(self):
        self.proc_init()

        while True:
            cmd_exit = False
            while True:
                try:
                    cmd = self.cmd_queue.get_nowait()
                    if cmd == 'exit':
                        cmd_exit = True
                except Empty:
                    break
            if cmd_exit:
                break
            
            try:
                job = self.job_queue.get_nowait()
                entry = self.proc_job_recieved(job)
                try:
                    if self.on_queue is not None:
                        self.on_queue(entry)
                    else:
                        self.result_queue.put_nowait(entry)
                except Full:
                    print('BaseUpscaler.proc_main: Result queue is full. Is consumer not fast enough?')
            except Empty:
                time.sleep(0.001)

        os.kill(os.getpid(), 0)

    def push_frame(self, entry, timeout=10):
        self.job_queue.put(entry, timeout=timeout)

    def get_frame(self, timeout=10):
        entry = self.result_queue.get(timeout=timeout)
        return entry

    def join(self, timeout=15):
        self.proc.join(timeout=timeout)
    
    def wait_for_job_clear(self):
        while not self.job_queue.empty():
            time.sleep(0.001)

    def stop(self):
        self.cmd_queue.put('exit')
        self.join()
    
    #@abs.abstractmethod
    def proc_init(self):
        pass

    #@abs.abstractmethod
    def proc_job_recieved(self, job):
        pass

    #@abs.abstractmethod
    def proc_cleanup(self):
        pass