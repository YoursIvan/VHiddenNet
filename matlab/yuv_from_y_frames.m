frames = 122;
row = 360;
col = 480;
yuvfile = 'temp.yuv';

[v_l_y,v_l_cb,v_l_cr] = readyuv(yuvfile,row,col,frames);


for i = 1 : frames
    Yimg =imread(['C:\Users\chendiao\Desktop\StegaStamp\frames_output\',num2str(i),'.jpg']);
    v_l_y(:,:,i) = double(Yimg(:,:,1));
end


saveyuv(v_l_y,v_l_cb,v_l_cr,row,col,frames,'stego_v.yuv')

%transcode
stegofile = 'stego_v.yuv';
tempfile = 'stego_v.mp4';
transfile = 'trans.yuv';

if(exist(tempfile,'file'))
    delete(tempfile);
end


command = sprintf('ffmpeg -y -s 480*360 -pix_fmt yuv420p -i %s -vcodec h264  -c:v libx264 -preset faster -crf 26 %s -v quiet',stegofile,tempfile);
system(command);
command = sprintf('ffmpeg -i %s -y %s -v quiet',tempfile,transfile);
system(command);

[v_l_y,v_l_cb,v_l_cr] = readyuv( transfile,row,col,frames);

for i = 1: frames
    img = v_l_y(:,:,i);
    imwrite(double(img)/255,['frames_trans/',num2str(i),'.jpg'],'quality',100);
end
 
 
