import threading
import io

class WithWrapper:
    def __init__(self, enter, exit) -> None:
        self.enter = enter
        self.exit = exit
    
    def __enter__(self):
        self.enter()
 
    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()

class ReaderWriterObject:
    def __init__(self, object) -> None:
        self.object = object
        self.mutex = threading.RLock()
        self.wrt = threading.RLock()
        self.nreader = 0
    
    def get(self): return self.object
    def set(self, v): self.object = v
    
    def read(self):
        return WithWrapper(self.start_read, self.end_read)
    
    def start_read(self):
        self.mutex.acquire()
        self.nreader += 1
        if (self.nreader == 1):
            self.wrt.acquire()
        self.mutex.release()
        
    def end_read(self):
        self.mutex.acquire()
        self.nreader -= 1
        assert self.nreader >= 0
        if (self.nreader == 0):
            self.wrt.release()
        self.mutex.release()
    
    def write(self):
        return WithWrapper(self.start_write, self.end_write)
    
    def start_write(self):
        self.wrt.acquire()
    
    def end_write(self):
        self.wrt.release()
        
class ImageCache:
    def has_file(self, filename:str):
        pass
    
    def read_file(self, filename:str):
        pass
    
    def write_file(self, filename:str, buffer:io.BytesIO):
        pass