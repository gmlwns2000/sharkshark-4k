import argparse
import os
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()
    
    assert args.dir is not None
    
    for file in sorted(os.listdir(args.dir)):
        assert len(file) > 4
        file_path = os.path.join(args.dir, file)
        upfile_path = os.path.join(args.dir, f"[SS4]{file[:-4]}.flv")
        if os.path.exists(upfile_path):
            print(f'PipelineFolder: skip', upfile_path)
            continue
        cmds = [
            f"python", "-m", "main.upscaler", 
            "--url", file_path, "--quality", "720p", "--no-frame-skips", 
            "--output-file", upfile_path, 
            "--hr-level", "0", "--lr-level", "3", "--fps", "24"
        ]
        print(f'PipelineFolder: cmds {cmds}')
        retcode = subprocess.call(cmds)
        print(f'PipelineFolder: Processed, retcode {retcode} for {file_path}')