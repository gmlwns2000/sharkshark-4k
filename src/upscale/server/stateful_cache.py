import io
import logging
import os
import random
import sys
import threading
import time
from typing import Dict, Tuple
from util import human_readable
import filelock

from upscale.server.image_cache import *

logger = logging.getLogger("StatefulImageCache")
logger.setLevel(logging.DEBUG)

class DiskImageCache(ImageCache):
    def __init__(self, max_size=1024*1024*1024*2) -> None:
        super().__init__()
        self.root = './cache'
        os.makedirs(self.root, exist_ok=True)
        
        self.idx = 0
        # this lock mechanism is not working...
        while True:
            lock_path = os.path.join(self.root, f'{self.idx}.lock')
            try:
                lock = filelock.FileLock(lock_path, timeout=0.3)
                time.sleep(random.random())
                if lock.is_locked:
                    self.idx += 1
                    continue
                lock.acquire()
                logger.debug(f'file lock {lock_path} acquired')
                break
            except filelock.Timeout:
                self.idx += 1
        
        self.path = os.path.join(self.root, f'{self.idx}_storage')
        os.makedirs(self.path, exist_ok=True)
        
        self.max_size = max_size
        self.lru_table = ReaderWriterObject({})
        self.size = 0
        
        self.load_cache()
        
        self.fit_cache(max_size)
    
    def get_path(self, filename):
        return os.path.join(self.path, filename)

    def load_cache(self):
        table = self.lru_table.get()
        for file in os.listdir(self.path):
            table[file] = time.time()
        self.update_size()
        
        logger.debug(f'loaded cache {self.size/self.max_size*100:.2f}% ({human_readable(self.size)})')
    
    def update_size(self):
        size = 0
        for file in os.listdir(self.path):
            try:
                size += os.path.getsize(self.get_path(file))
                with self.lru_table.write():
                    if not file in self.lru_table.get(): # just created
                        self.lru_table.get()[file] = time.time()
            except FileNotFoundError:
                pass
        self.size = size
        return size
    
    def fit_cache(self, new_size):
        if self.size < new_size: return
        self.update_size()
        
        with self.lru_table.read():
            table = self.lru_table.get()
            files = os.listdir(self.path)
            files = sorted(zip(files, [table[file] if file in table else 0 for file in files]), key=lambda it: it[1])
            
            to_evict = self.size - new_size
            while to_evict > 0 and len(files) > 0 and self.size > new_size:
                lru_file = files.pop(0)[0]
                file_path = self.get_path(lru_file)
                if os.path.exists(file_path):
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        self.size -= file_size
                        to_evict -= file_size
                        logger.debug(f'removed {lru_file} cache {self.size/self.max_size*100:.2f}% ({human_readable(self.size)})')
                    except FileNotFoundError:
                        self.update_size()
                        to_evict = self.size - new_size
                else:
                    self.update_size()
                    to_evict = self.size - new_size
                
        self.update_size()
    
    def has_file(self, filename):
        if os.path.exists(self.get_path(filename)):
            with self.lru_table.write():
                self.lru_table.get()[filename] = time.time()
            return f'/upscale/file/{filename}'
        else:
            return None
    
    def read_file(self, filename):
        file_path = self.get_path(filename)
        if os.path.exists(file_path):
            with self.lru_table.write():
                self.lru_table.get()[filename] = time.time()
            try:
                with open(file_path, 'rb') as f:
                    return io.BytesIO(f.read())
            except FileNotFoundError:
                return None
        else:
            return None
    
    def write_file(self, filename, buffer):
        file_path = self.get_path(filename)
        with self.lru_table.write():
            if isinstance(buffer, io.BytesIO):
                buffer = buffer.getbuffer()
                size = len(buffer)
            elif isinstance(buffer, bytes):
                size = len(buffer)
            
            self.fit_cache(self.max_size - size)
            
            with open(file_path, 'wb') as f:
                logger.debug(f'writed {filename} cache {self.size/self.max_size*100:.2f}% ({human_readable(self.size)})')
                f.write(buffer)
                self.lru_table.get()[filename] = time.time()
        
class MemoryImageCache:
    def __init__(self, cache_size_bytes=1024*1024*1024*2) -> None:
        self.max_size = cache_size_bytes #1020*1024*8
        self.size = 0
        self._bank: Dict[str, Tuple[float, ReaderWriterObject]] = {} #filename -> (last_tick, mutex, wrt, num_readers, io.BytesIO)
        self.bank = ReaderWriterObject(self._bank)
    
    def contains(self, key):
        with self.bank.read():
            has_key = key in self.bank.get()
            # logger.debug(f'contains {key} {has_key}')
            return has_key
    
    def sizeof(self, key):
        with self.bank.read():
            if not self.contains(key): return 0
            item = self.bank.get()[key][-1] #type: ReaderWriterObject
            with item.read():
                buffer = item.get() #type: io.BytesIO
                return sys.getsizeof(buffer)
    
    def remove(self, key):
        if not self.contains(key): return
        with self.bank.write():
            size = self.sizeof(key)
            self.size -= size
            item = self.bank.get()[key]
            with item[-1].write():
                del self.bank.get()[key]
        
        logger.debug(f'removed {self.size / self.max_size * 100: .5f}% ({human_readable(self.size)} in cache)')
    
    def add(self, key, value: io.BytesIO):
        if self.contains(key):
            #raise Exception('already exists')
            logger.debug(f'memcache already exists {key}')
            return
        
        with self.bank.write():
            self.size += sys.getsizeof(value)
            if self.size > self.max_size:
                logger.debug(f'need to evict {self.size} {self.max_size}')
                self.evict(self.size - self.max_size)
            
            self.bank.get()[key] = (time.time(), ReaderWriterObject(value))
        
        logger.debug(f'added {self.size / self.max_size * 100: .5f}% ({human_readable(self.size)} in cache)')
    
    def cloneof(self, key):
        with self.bank.read():
            if not self.contains(key): raise Exception('not found')
            item = self.bank.get()[key]
            with item[-1].read():
                self.bank.get()[key] = (time.time(), item[-1])
                buffer = item[-1].get() #type: io.BytesIO
                return io.BytesIO(buffer.getvalue())
    
    def evict(self, to_bytes):
        #threadsafe here
        logger.debug(f'need to evict {to_bytes} bytes')
        while to_bytes > 0 and len(self.bank.get()) > 0:
            minval = 987654312340
            minidx = None
            for key, value in self.bank.get().items():
                if value[0] < minval:
                    minval = value[0]
                    minidx = key
            if minidx is not None:
                size = self.sizeof(minidx)
                to_bytes -= size
                logger.debug(f'evict {size} bytes')
                self.remove(minidx)
            else:
                return
    
    # interfaces
    def has_file(self, filename:str):
        if self.contains(filename):
            return f'/upscale/file/{filename}'
        else: return None
    
    def read_file(self, filename:str):
        if self.contains(filename):
            return self.cloneof(filename)
        else:
            return None
    
    def write_file(self, filename:str, buffer:io.BytesIO):
        self.add(filename, buffer)