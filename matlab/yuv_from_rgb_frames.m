frames = 122;
yuvfile = '1_768x576.yuv';

[v_l_y,v_l_cb,v_l_cr] = readyuv(yuvfile,576,768,frames);


for i = 1 : frames
    YUVimg = rgb2ycbcr(imread(['C:\Users\chendiao\Desktop\StegaStamp\frames_output\',num2str(i),'.jpg']));
    [imgHeight imgWidth imgDim] = size(YUVimg);         %%
    len = imgHeight*imgWidth*imgDim;
    yuvimout = zeros(1,len);
    Y = YUVimg(:,:,1);     % Y 矩阵
    U = YUVimg(:,:,2);     % U 矩阵
    V = YUVimg(:,:,3);     % V 矩阵
    yuv420sampY = Y;
    yuv420sampU = U(1:2:size(U,1),1:2:size(U,2));
    yuv420sampV = V(2:2:size(V,1),1:2:size(V,2));
    
    
    v_l_y(:,:,i) = double(yuv420sampY);
    v_l_cb(:,:,i) = double(yuv420sampU);
    v_l_cr(:,:,i) = double(yuv420sampV);
end


saveyuv(v_l_y,v_l_cb,v_l_cr,576,768,frames,'stego_v.yuv')

%transcode
stegofile = 'stego_v.yuv';
tempfile = 'stego_v.mp4';
transfile = 'trans.yuv';

command = sprintf('ffmpeg -y -s 768*576 -pix_fmt yuv420p -i %s -vcodec h264  -c:v libx264 -preset faster -crf 28 %s -v quiet',stegofile,tempfile);
system(command);
command = sprintf('ffmpeg -i %s -y %s -v quiet',tempfile,transfile);
system(command);

coverfile = 'stego_v.mp4';
obj = VideoReader(coverfile);%输入视频位置
numFrames = obj.NumberOfFrames;% 帧的总数
 for k = 1 : numFrames% 读取前15帧
     frame = read(obj,k);%读取第几帧
    %imshow(frame);%显示帧
     imwrite(frame,strcat('frames_trans\',num2str(k),'.jpg'),'jpg');% 保存帧
 end
