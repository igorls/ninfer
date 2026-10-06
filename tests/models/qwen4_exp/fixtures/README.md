# Synthetic video fixture

`red-blue.mp4` contains two seconds of red followed by two seconds of blue, 112 by 112 pixels,
two frames per second, H.264/yuv420p, without audio. It is generated test data, with no external
media source. Reproduce with FFmpeg:

```sh
ffmpeg -f lavfi -i 'color=c=red:s=112x112:r=2:d=2' -f lavfi -i 'color=c=blue:s=112x112:r=2:d=2' -filter_complex '[0:v][1:v]concat=n=2:v=1:a=0[v]' -map '[v]' -c:v libx264 -pix_fmt yuv420p -movflags +faststart red-blue.mp4
```
