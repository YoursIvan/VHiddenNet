%Data Augment
%flipped

srcpath = 'D:\Data\Boss_jpg\95\';
dstpath = 'D:\Data\Boss_jpg\flipped\';

for i = 1 : 10000
    i
    srcfile = [srcpath,num2str(i),'.jpg'];
    dst1file = [dstpath,num2str(10000+i),'.jpg'];
    dst2file = [dstpath,num2str(20000+i),'.jpg'];
    dst3file = [dstpath,num2str(30000+i),'.jpg'];
    I=imread(srcfile); %输入图像
    J1=flip(I,1);%原图像的水平镜像
    J2=flip(I,2);%原图像的垂直镜像
    J3=flip(I,3);%原图像的水平垂直镜像
    imwrite(J1,dst1file,'quality',95);
    imwrite(J2,dst2file,'quality',95);
    imwrite(J3,dst3file,'quality',95);
end