import tqdm
import os, sys, requests, logging
import time
import multiprocess as mp
from multiprocessing import Pool

logger = logging.getLogger("Test.ImagePipeline")

IMNET_PATH = '/d1/dataset/ILSVRC2012'
TARGET_SERVER = 'https://ainl.tk'

TEST_REQUEST_FILE = True

def test_file(file):
    cache = err = ok = 0
    with open(file, 'rb') as f:
        res = requests.request(
            'POST', url=f'{TARGET_SERVER}/upscale/image', files={'file': f}
        )
        if res.status_code == 200:
            data = res.json()
            if data['result'] == 'ok':
                if 'cache' in data:
                    cache = 1
                if TEST_REQUEST_FILE:
                    res = requests.request('GET', url=f'{TARGET_SERVER}{data["url"]}')
                    if res.status_code == 200:
                        try:
                            for chunk in res.iter_content(chunk_size=32*1024):
                                pass
                            ok = 1
                        except:
                            err = 1
                    else:
                        err = 1
                else:
                    ok = 1
            else:
                err = 1
        else:
            err = 1
    return {
        'ok': ok,
        'err': err,
        'cache': cache,
    }

def test_files(files):
    results = []
    t = time.time()
    # for file in tqdm.tqdm(files):
    #     results.append(test_file(file))
    with Pool(512) as pool:
        # results = pool.map(test_file, files)
        with tqdm.tqdm(total=len(files)) as pbar:
            for result in pool.imap_unordered(test_file, files):
                results.append(result)
                pbar.update()
    elapsed = time.time() - t

    caches = errs = oks = 0
    for result in results:
        caches += result['cache']
        errs += result['err']
        oks += result['ok']

    return {
        'ok_rate':oks / len(files),
        'err_rate':errs / len(files),
        'cache_rate':caches / len(files),
        'rps':len(files) / elapsed,
    }

def test_long(files):
    results = []
    for i in range(5):
        results.append(test_files(files))
    print(results)

if __name__ == '__main__':
    files = []
    dir_path = f'{IMNET_PATH}/test'
    for file in os.listdir(dir_path):
        files.append(os.path.join(dir_path, file))
    files = files[:1000]
    test_long(files)