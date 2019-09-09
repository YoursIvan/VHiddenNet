%extract Y from yuv
row = 360;
col = 480;
frames = 122;
filename = 'C:\Users\chendiao\Desktop\StegaStamp\Samples\temp\105_1920x1080.mp4';

yuvfile = 'temp.yuv';
command = sprintf('ffmpeg -i %s -y %s -v quiet',filename,yuvfile);
system(command);

[v_l_y,v_l_cb,v_l_cr] = readyuv( yuvfile,row,col,frames);

for i = 1: frames
    img = v_l_y(:,:,i);
    imwrite(double(img)/255,['frames/',num2str(i),'.jpg']);
end