% for i = 1:50
%     mkdir (num2str(i))
% end


row = 360;
col = 480;
frames = 190;
yuvfile = 'temp.yuv';

for i = 1: 100
    filename = ['C:\Users\chendiao\Desktop\StegaStamp\Samples\temp\',num2str(i),'_1920x1080.mp4'];
    command = sprintf('ffmpeg -i %s -y %s -v quiet',filename,yuvfile);
    system(command);

    [v_l_y,v_l_cb,v_l_cr] = readyuv( yuvfile,row,col,frames);

    for j = 1: frames
        count = frames*(i-1)+j
        img = v_l_y(:,:,j);
        imwrite(double(img)/255,['C:\Users\chendiao\Desktop\StegaStamp\Samples\frame_set\480p\',num2str(count),'.jpg']);
    end
end