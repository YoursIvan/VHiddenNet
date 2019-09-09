clear all; clc;
cover_name = '1.mp4';
stego_name = 'stego_v.mp4';
cover_yuv = '1.yuv';
stego_yuv = 'stego_v.yuv';
%48.25
%47.6613

%0.3105
%0.2697

RoR = 0;
command = sprintf('ffmpeg -i %s -y %s -v quiet',cover_name,cover_yuv);
system(command);

command = sprintf('ffmpeg -i %s -y %s -v quiet',stego_name,stego_yuv);
system(command);


cover = fopen(cover_yuv ,'r'); %读入文件
stego = fopen(stego_yuv ,'r');
row= 360;col=480; %图像的高、宽
frames=122; % total=97 %序列的帧数

PSNRTotal = 0;

for frame=1:frames
 %读入文件 将yuv转换为rgb，并用imshow显示
  %  im_l_y=fread(fid,[row,col],'uchar');  %错误的读入
    im_l_y = zeros(row,col); %Y
    for i1 = 1:row 
       im_l_y(i1,:) = fread(cover,col);  %读取数据到矩阵中 
    end

    im_l_cb = zeros(row/2,col/2); %cb
    for i2 = 1:row/2 
       im_l_cb(i2,:) = fread(cover,col/2);  
    end

    im_l_cr = zeros(row/2,col/2); %cr
    for i3 = 1:row/2 
       im_l_cr(i3,:) = fread(cover,col/2);  
    end

    %由于输入的yuv文件为4:2:0，所以CbCr要改变大小，
    %否则im_l_ycbcr(:, :, 2) =im_l_cb;会出现错误
    im_l_cb = imresize(im_l_cb, [row, col], 'bicubic');%改变图像的大小
    im_l_cr = imresize(im_l_cr, [row, col], 'bicubic');
    im_c_ycbcr = zeros([row, col, 3]);
    im_c_ycbcr(:, :, 1) = im_l_y;
    im_c_ycbcr(:, :, 2) = im_l_cb;
    im_c_ycbcr(:, :, 3) = im_l_cr;

 %读入文件 将yuv转换为rgb，并用imshow显示
  %  im_l_y=fread(fid,[row,col],'uchar');  %错误的读入
    for i1 = 1:row 
       im_l_y(i1,:) = fread(stego,col);  %读取数据到矩阵中 
    end

    im_l_cb = zeros(row/2,col/2); %cb
    for i2 = 1:row/2 
       im_l_cb(i2,:) = fread(stego,col/2);  
    end

    im_l_cr = zeros(row/2,col/2); %cr
    for i3 = 1:row/2 
       im_l_cr(i3,:) = fread(stego,col/2);  
    end

    %由于输入的yuv文件为4:2:0，所以CbCr要改变大小，
    %否则im_l_ycbcr(:, :, 2) =im_l_cb;会出现错误
    im_l_cb = imresize(im_l_cb, [row, col], 'bicubic');%改变图像的大小
    im_l_cr = imresize(im_l_cr, [row, col], 'bicubic');
    im_s_ycbcr = zeros([row, col, 3]);
    im_s_ycbcr(:, :, 1) = im_l_y;
    im_s_ycbcr(:, :, 2) = im_l_cb;
    im_s_ycbcr(:, :, 3) = im_l_cr;
    
    MSE = double(sum(sum((im_s_ycbcr(:, :, 1) - im_c_ycbcr(:, :, 1)).^2)))/(row*col);
    if(MSE==0)
        PSNR =0;
    else
        PSNR = 20*log10(max(max(im_c_ycbcr(:, :, 1)))/sqrt(MSE));
    end
    PSNRTotal = PSNRTotal+PSNR;
end
 
flash = 0;
[v_l_y1,v_l_cb1,v_l_cr1] = readyuv( cover_yuv,row,col,frames);
[v_l_y2,v_l_cb2,v_l_cr2] = readyuv( stego_yuv,row,col,frames);
for i = 2:frames-1
    t = abs(v_l_y2(:,:,i+1) - v_l_y2(:,:,i)) + abs(v_l_y2(:,:,i-1) - v_l_y2(:,:,i));
    %t2 = abs(v_l_y2(:,:,i+1) - v_l_y2(:,:,i))+ abs(v_l_y2(:,:,i-1) - v_l_y2(:,:,i));
    
    r1 = abs(v_l_y1(:,:,i) - v_l_y2(:,:,i))- (v_l_y1(:,:,i-1) - v_l_y2(:,:,i-1));
    r2 = abs(v_l_y1(:,:,i) - v_l_y2(:,:,i))- (v_l_y1(:,:,i+1) - v_l_y2(:,:,i+1));
    RoR = RoR + sum(sum(r1+r2))/(row*col);
    flash = flash + sum(sum(t))/(row*col);
end
flash = flash/(frames-2)
RoR = RoR/(frames-2)
Video_PSNR = PSNRTotal/frames