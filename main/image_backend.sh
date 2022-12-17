while true
do
gunicorn -w 1 --threads 256 --bind 0.0.0.0:8087 upscale.server.image_pipeline:app
echo [ERROR] SERVER CRASHED. SLEEP 3 SEC
sleep 3
done
