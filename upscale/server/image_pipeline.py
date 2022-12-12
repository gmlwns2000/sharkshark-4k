import os
import threading
import flask as fl
import multiprocessing as mp
import logging
import torch, time, cv2
import numpy as np
from upscale.fsrcnn_upscaler import FsrcnnUpscalerService, UpscalerQueueEntry
from util.profiler import Profiler
import queue

logger = logging.getLogger("ImagePipeline")
logger.setLevel(logging.DEBUG)
blueprint = fl.Blueprint('upscale', __name__, url_prefix='/upscale')

upscaler_lock = mp.Lock()
upscaler = None
upscaler_event = mp.Event()

last_id_lock = mp.Lock()
last_id = 0

def get_id():
    global last_id
    with last_id_lock:
        new_id = last_id
        last_id += 1
    return new_id

def get_pipeline():
    global upscaler
    with upscaler_lock:
        if upscaler is None:
            upscaler = FsrcnnUpscalerService(
                lr_level=3, device=0, denoising=False, denoise_rate=0.5, 
                upscaler_model='realesrgan', batch_size=1, jit_mode=False, lr_hr_resize=False
            )
            upscaler.start()
            logging.info('Upscaler started')
    return upscaler

@blueprint.route('/ping')
def ping():
    return 'pong'

@blueprint.route('/file/<filename>')
def upload_file(filename):
    logger.debug(filename)
    logger.debug(f'{blueprint.root_path}')
    return fl.send_from_directory(os.getcwd(), os.path.join('cache', filename))

def write_file(mid, img):
    os.makedirs('./cache', exist_ok=True)
    cv2.imwrite(f'./cache/{mid}.png', img)
    return f'/upscale/file/{mid}.png'

@blueprint.route('/image', methods=['POST'])
def upscale_image():
    profiler = Profiler()
    profiler.start('endpoint.pipeline')
    upscaler = get_pipeline()
    profiler.end('endpoint.pipeline')
    
    profiler.start('endpoint.io')
    my_id = get_id()
    buffer = fl.request.files['file'].read()
    buffer = np.frombuffer(buffer, dtype=np.uint8)
    profiler.start('endpoint.io.imdecode')
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    profiler.end('endpoint.io.imdecode')
    profiler.end('endpoint.io')
    
    if img is None:
        return fl.jsonify({
            'result':'err', 
            'err': 'img is none. did you give correct image blob?',
            'profiler': entry.profiler.data
        })
    
    if img.shape[-1] != 3:
        return fl.jsonify({
            'result':'err', 
            'err': 'img must be RGB, 3 channel',
            'profiler': entry.profiler.data
        })
    
    assert img.shape[-1] == 3
    logger.debug(img.shape)
    
    profiler.start('endpoint.proc')
    upscaler.push_job(UpscalerQueueEntry(
        frames=torch.tensor(img, dtype=torch.uint8, device=upscaler.device).unsqueeze(0),
        audio_segment=torch.empty((1,)),
        step=my_id,
        elapsed=0,
        last_modified=time.time(),
        profiler=profiler,
    ))
    
    #wait for id is finished
    while True:
        try:
            entry = upscaler.get_result() #type: UpscalerQueueEntry
            logger.debug(f'processed id {entry.step}')
            if entry.step == my_id:
                profiler = entry.profiler
                break
            else:
                upscaler.result_queue.put(entry)
                upscaler_event.set()
            upscaler_event.wait(timeout=0.5)
            upscaler_event.clear()
        except TimeoutError:
            pass
    profiler.end('endpoint.proc')
    
    profiler.start('endpoint.write')
    if(entry.frames.device != 'cpu'):
        entry.frames = entry.frames.cpu()
    frames = entry.frames.squeeze(0).numpy()
    img_path = write_file(my_id, frames)
    logger.debug(f'{frames.shape} {img_path}')
    profiler.end('endpoint.write')
    
    return fl.jsonify({
        'result':'ok', 
        'url':img_path,
        'profiler': entry.profiler.data
    })

if __name__ == '__main__':
    app = fl.Flask(__name__)
    app.register_blueprint(blueprint)
    
    app.run(debug=True, port=8088, use_reloader=False, threaded=True, host='0.0.0.0')
