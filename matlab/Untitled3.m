%convert to gray
frames = 122;
for i = 1 : frames
    img = rgb2gray(imread(['C:\Users\chendiao\Desktop\douyin_Demo\test_transcode\frames\',num2str(i),'.jpg']));
    imwrite(double(img)/255,['C:\Users\chendiao\Desktop\douyin_Demo\test_transcode\frames_gray\',num2str(i),'.jpg']);
end