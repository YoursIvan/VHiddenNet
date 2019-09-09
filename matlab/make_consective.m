%make examples of consective three frames
row = 576;
col = 768;
frames = 190;
count = 190 * 104;
index = 1;
dst = 'D:\Data\VHiddenData\train_set\';
p_dir = 'D:\Data\VHiddenData\p_dis\';
m_dir = 'D:\Data\VHiddenData\m_dis\';

for i = 1: 104
    for j = 2: frames-1
        if(~exist([dst,num2str(index)],'dir'))
            mkdir([dst,num2str(index)]);
        end
        dst_dir = [dst,num2str(index),'\'];
        index
        for k = 1 : 3
            count = (i-1) * frames + j + (k-2);
            file_path = ['C:\Users\chendiao\Desktop\StegaStamp\Samples\frame_set\all_set\',num2str(count),'.jpg'];
            dst_path = [dst_dir,num2str(k),'.jpg'];
            %copyfile(file_path,dst_path)
            if(k == 2)
                cover = double(imread(file_path));
                [rhoP,rhoM] = S_UNIWARD(cover);
                save([p_dir,'rhoP_',num2str(index),'.mat'],'rhoP')
                save([m_dir,'rhoM_',num2str(index),'.mat'],'rhoM')
            end
        end
        index = index +1;
    end
end
    
