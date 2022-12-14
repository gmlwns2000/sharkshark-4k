gunicorn -w 2 --threads 32 --bind 0.0.0.0:8087 upscale.server.image_pipeline:app
