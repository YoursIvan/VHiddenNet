
path = 'C:\Users\chendiao\Desktop\frames\';
p_dir = 'C:\Users\chendiao\Desktop\p_dis\';
m_dir = 'C:\Users\chendiao\Desktop\m_dis\';

for i = 1:122
    i
    file_path = [path,num2str(i), '.jpg'];
    cover = double(imread(file_path));
    [rhoP,rhoM] = S_UNIWARD(cover);
    save([p_dir,'rhoP_',num2str(i),'.mat'],'rhoP')
    save([m_dir,'rhoM_',num2str(i),'.mat'],'rhoM')
end