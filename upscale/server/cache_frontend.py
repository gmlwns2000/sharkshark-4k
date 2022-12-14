import hashlib, httpx
import io

import redis
from upscale.server.stateless_cache import RedisImageCache
from fastapi import FastAPI, APIRouter, Response, UploadFile, status, HTTPException

router = APIRouter()
image_cache = RedisImageCache()

def get_bytes_hash(buffer):
    return hashlib.sha1(buffer).hexdigest()

def get_filename(buffer):
    return f'{get_bytes_hash(buffer)}.png'

ENDPOINT_UPSCALE = 'http://127.0.0.1:8087/upscale/image'

@router.post("/image")
async def upscale_image(file: UploadFile):
    lr_buffer = await file.read()
    filename = get_filename(lr_buffer)
    
    is_cached = await image_cache.has_file(filename)
    if is_cached:
        return {
            'result':'ok',
            'cache': 'hit',
            'url': f'/upscale/file/{filename}',
        }
    
    lock = image_cache.file_lock(filename, timeout=120, blocking_timeout=10)
    await lock.acquire(blocking=True)
    
    try:
        is_cached = await image_cache.has_file(filename)
        if is_cached:
            return {
                'result':'ok',
                'cache': 'hit',
                'url': f'/upscale/file/{filename}',
            }
        
        hr_buffer = None
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    ENDPOINT_UPSCALE, 
                    files = {'file': lr_buffer},
                    params = {'return_type': 'file'},
                    timeout = 100,
                )
                if response.status_code == 500:
                    return HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=response.text,
                    )
                elif response.status_code == 200:
                    hr_buffer = await response.aread()
                else:
                    raise Exception(response)
            except httpx.TimeoutException:
                return HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={'status':'err', 'err':'timeout request upscale/image'},
                )
            except httpx.ConnectError:
                return HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={'status':'err', 'err':'gpu server dead?'},
                )
        
        assert not hr_buffer is None
        
        await image_cache.write_file(filename, io.BytesIO(hr_buffer))
        
        return {
            'result':'ok',
            'cache': 'miss',
            'url': f'/upscale/file/{filename}',
        }
    finally:
        try:
            await lock.release()
        except redis.exceptions.LockError:
            pass
        except redis.exceptions.LockNotOwnedError:
            pass

@router.get("/file/{filename}")
async def download_file(filename: str):
    global image_cache
    buffer = await image_cache.read_file(filename)
    if buffer is not None:
        return Response(
            content=buffer, media_type="image/png", 
            headers={
                'Access-Control-Allow-Origin': '*'
            }
        )
    else:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={'result':'err', 'err':'not found'}
        )

app = FastAPI()
app.include_router(router, prefix='/upscale')