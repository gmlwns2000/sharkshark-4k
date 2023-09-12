import argparse
from ..sharkshark.pipeline import TwitchUpscalerPostStreamer
from ..stream.recoder import TW_VIICHAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
"""SharkShark4k - Web Video Upscaling Pipeline

-- Example --
# For live stream, this will restream the video into rtmp://localhost:1935/live
python -m main.upscaler --url https://www.twitch.tv/tizmtizm

# For twitch archive upscale. Output file should be eneded with .flv
python -m main.upscaler --url https://www.twitch.tv/videos/1663212121 --no-frame-skips --output-file output.flv

# For fix audio sync. When audio_queue > 0, the audio is delayed about `audio_queue` seconds. Should be integer
python -m main.upscaler --url https://www.twitch.tv/tizmtizm --audio-queue 1

# For process local file, FLV file should be played in VLC
python -m main.upscaler --url mashup.mp4 --quality 1080p --no-frame-skips --output-file output.flv --hr-level 2 --lr-level 5 --fps 32
""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--url', type=str, default=TW_VIICHAN, help='default=GORANI')
    parser.add_argument('--quality', type=str, default='1080p60', help='ex: 1080p60, 720p60, 360p, etc. depending on providers')
    parser.add_argument('--fps', type=int, default=24, help='default=24')
    parser.add_argument('--denoise-rate', type=float, default=0.75, help='in Real[0,1], default=0.75')
    parser.add_argument('--hr-level', type=int, default=0, help='Output scale: 1440p[default], 1800p, 2160p')
    parser.add_argument('--lr-level', type=int, default=3, help='Input processing scale: 360p, 540p, 630p, 720p[default], 900p, 1080p')
    parser.add_argument('--audio-queue', type=int, default=0, help='0 for no delay, N for N sec audio delay insert.')
    parser.add_argument('--output-file', type=str, default='rtmp://127.0.0.1:1935/live', help='default=rtmp://127.0.0.1:1935/live')
    parser.add_argument('--no-frame-skips', action='store_true', default=False, help='prevent frame skip. for static file conversion')

    args = parser.parse_args()
    print(args)
    
    pipeline = TwitchUpscalerPostStreamer(
        url = args.url, fps = args.fps, denoising=False, lr_level=args.lr_level, quality=args.quality, 
        frame_skips=not args.no_frame_skips, denoise_rate=args.denoise_rate, hr_level=args.hr_level,
        output_file=args.output_file, audio_skip=args.audio_queue,
    )
    
    pipeline.start()
    pipeline.join()