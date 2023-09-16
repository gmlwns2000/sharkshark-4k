import abc, time, os
import signal
from queue import Empty, Full
import traceback
import torch.multiprocessing as mp

class ProcessDeadException(Exception):
    pass

class BaseService(metaclass=abc.ABCMeta):
    on_queue = None
    exit_on_error = False

    def __init__(self) -> None:
        self.job_queue = mp.Queue(maxsize=32)
        self.result_queue = mp.Queue(maxsize=32)
        self.cmd_queue = mp.Queue(maxsize=4096)
        self.proc = mp.Process(target=self.proc_pre_main, daemon=True)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        if 'proc' in state:
            del state["proc"]
        return state

    def start(self):
        self.proc.start()

    def proc_pre_main(self):
        self.proc_main()

    def proc_main(self):
        try:
            self.proc_init()

            alive = True
            while alive:
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
                            alive = self.on_queue(entry)
                            assert alive is not None
                        else:
                            self.result_queue.put_nowait(entry)
                    except Full:
                        print('BaseUpscaler.proc_main: Result queue is full. Is consumer not fast enough?')
                except Empty:
                    time.sleep(0.001)

            print('BaseService.proc_main: EXIT PATTERN!')
            # os.kill(os.getpid(), 15)
        except Exception as ex:
            if self.exit_on_error:
                traceback.print_exc()
                print(ex)
                os.killpg(os.getpgid(os.getpid()), signal.SIGINT)
            else:
                raise ex

    def check_proc(self):
        if not self.exit_on_error: return
        # if self.proc is not None:
        #     raise Exception('process is not started')
        if not self.proc.is_alive():
            if self.exit_on_error:
                try:
                    raise ProcessDeadException('process is dead!')
                except Exception as ex:
                    traceback.print_exc()
                    print(ex)
                    os.killpg(os.getpgid(os.getpid()), signal.SIGINT)
            else:
                raise ProcessDeadException('process is dead!')
    
    def push_job(self, entry, timeout=10):
        self.check_proc()
        self.job_queue.put(entry, timeout=timeout)
    
    def push_job_nowait(self, entry):
        self.check_proc()
        self.job_queue.put_nowait(entry)

    def get_result(self, timeout=10):
        self.check_proc()
        entry = self.result_queue.get(timeout=timeout)
        return entry

    def join(self, timeout=15):
        self.proc.join(timeout=timeout)
        return self.proc.exitcode
    
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