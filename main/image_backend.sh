gunicorn -w 1 --threads 256 --bind 0.0.0.0:8088 upscale.server.image_pipeline:app
