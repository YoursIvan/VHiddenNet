function [v_l_y,v_l_cb,v_l_cr] = readyuv( filename,row,col,frames)
    %UNTITLED 此处显示有关此函数的摘要
    %   此处显示详细说明
    f1 = fopen(filename,'r'); %读入文件
    v_l_y = zeros(row,col,frames);
    v_l_cb = zeros(row/2,col/2,frames);
    v_l_cr = zeros(row/2,col/2,frames);
    for frame=1:frames
     %读入文件 将yuv转换为rgb，并用imshow显示
      %  im_l_y=fread(fid,[row,col],'uchar');  %错误的读入
        im_l_y = zeros(row,col); %Y
        for i1 = 1:row 
           im_l_y(i1,:) = fread(f1,col);  %读取数据到矩阵中 
        end

        im_l_cb = zeros(row/2,col/2); %cb
        for i2 = 1:row/2 
           im_l_cb(i2,:) = fread(f1,col/2);  
        end

        im_l_cr = zeros(row/2,col/2); %cr
        for i3 = 1:row/2 
           im_l_cr(i3,:) = fread(f1,col/2);  
        end

        %由于输入的yuv文件为4:2:0，所以CbCr要改变大小，
        %否则im_l_ycbcr(:, :, 2) =im_l_cb;会出现错误
        
        v_l_y(:, :,frame) = im_l_y;
        v_l_cb(:, :,frame) = im_l_cb;
        v_l_cr(:, :,frame) = im_l_cr;
        
    end
    fclose(f1);
end

