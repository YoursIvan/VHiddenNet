%transcode
stegofile = 'stego_v.yuv';
tempfile = 'stego_v.mp4';
transfile = 'trans.yuv';

command = sprintf('ffmpeg -s 768*576 -pix_fmt yuv420p -i %s -vcodec h264 -qscale 10 %s -v quiet',stegofile,tempfile);
system(command);
command = sprintf('ffmpeg -i %s -y %s -v quiet',tempfile,transfile);
system(command);