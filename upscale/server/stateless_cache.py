import io
from typing import Union

import redis.asyncio as redis
from upscale.server.image_cache import *

class LockableImageCache(ImageCache):
    def lock_file(self, filename, timeout=0):
        pass

class RedisImageCache(ImageCache):
    def __init__(self, host='localhost', port=6379, prefix='ss4_'):
        self.prefix = prefix
        self.lock_suffix = '_lock'
        self.rd = redis.Redis(host=host, port=port, db=0)
    
    def file_lock(self, filename, timeout=30, blocking_timeout=10):
        return self.rd.lock(
            self.prefix + filename + self.lock_suffix, 
            timeout=timeout, 
            blocking_timeout=blocking_timeout
        )
    
    async def reset(self):
        await self.rd.flushdb()
    
    async def has_file(self, filename:str):
        return (await self.rd.exists(self.prefix+filename)) > 0
    
    async def read_file(self, filename:str) -> Union[None, bytes]:
        return await self.rd.get(self.prefix+filename)
    
    async def write_file(self, filename:str, buffer:io.BytesIO):
        await self.rd.set(self.prefix+filename, buffer.getbuffer())

if __name__ == '__main__':
    cache = RedisImageCache()
    cache.reset()
    
    fname = '1'
    assert cache.read_file(fname) == None
    assert cache.has_file(fname) == False
    buffer = b'1234'
    cache.write_file(fname, io.BytesIO(buffer))
    assert cache.has_file(fname) == True
    assert cache.read_file(fname) == buffer
    # cache.reset()
    # assert cache.read_file(fname) == None
    # assert cache.has_file(fname) == False