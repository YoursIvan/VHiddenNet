coverfile = 'trans.mp4';
yuvfile = 'temp.yuv';
command = sprintf('ffmpeg -i %s -y %s -v quiet',coverfile,yuvfile);
system(command);

[v_l_y,v_l_cb,v_l_cr] = readyuv( yuvfile,row,col,frames);

for i = 1: frames
    img = v_l_y(:,:,i);
    imwrite(double(img)/255,['frames_trans/',num2str(i),'.jpg']);
end
