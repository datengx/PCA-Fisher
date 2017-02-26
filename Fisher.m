%% Fisher face gender discrimination
%% Name:Da Teng
%% NOTE assumeing the face_data directory to be in the same directory
%% of this file
% Reading male images
curr_dir = pwd;
dir_name = strcat(curr_dir, '/face_data/male_face');
lm_dir_name = strcat(curr_dir, '/face_data/male_landmark_87');
start_idx = 0;
num_male_imgs = 88;
num_male_training = 78;
h = 256;
w = 256;
img_array_male = zeros(h*w, num_male_imgs);
lm_array_male = zeros(174, num_male_imgs);
idx = 0;
for i = start_idx:(start_idx + num_male_imgs -1)
   if i ~= 57
       image_name = sprintf('face%03d.bmp', i);
       full_path = strcat(dir_name, '/', image_name);
       % disp(full_path);
       img = imread(full_path);
       lm_name = sprintf('face%03d_87pt.txt', i);
       full_path = strcat(lm_dir_name, '/', lm_name);
       lm = importdata(full_path);
       lm = [lm(1:87)' lm(88:174)'];
       lm = reshape(lm', [174, 1]);
       lm = 256 - lm;
       lm = lm(1:end); % rid of the first element
       img_array_male(:,idx+1) = reshape(img, [h*w,1]);
       lm_array_male(:,idx+1) = lm';
       idx = idx + 1;
   end
end
% Reading female images
dir_name = strcat(curr_dir, '/face_data/female_face');
lm_dir_name = strcat(curr_dir, '/face_data/female_landmark_87');
start_idx = 0;
num_female_imgs = 85;
num_female_training = 75;
img_array_female = zeros(h*w, num_female_imgs);
lm_array_female = zeros(174, num_female_imgs);
for i = start_idx:(start_idx + num_female_imgs -1)
   image_name = sprintf('face%03d.bmp', i);
   full_path = strcat(dir_name, '/', image_name);
   % disp(full_path);
   img = imread(full_path);
   lm_name = sprintf('face%03d_87pt.txt', i);
   full_path = strcat(lm_dir_name, '/', lm_name);
   lm = importdata(full_path);
   lm = lm(1:end); % rid of the first element
   lm = [lm(1:87)' lm(88:174)'];
   lm = reshape(lm', [174, 1]);
   lm = 256 - lm;
   img_array_female(:,i+1) = reshape(img, [h*w,1]);
   lm_array_female(:,i+1) = lm';
end
% Reading unknown images
dir_name = strcat(curr_dir, '/face_data/unknown_face');
start_idx = 0;
num_imgs = 4;
img_array_unknown = zeros(h*w, num_imgs);
for i = start_idx:(start_idx + num_imgs -1)
   image_name = sprintf('face%03d.bmp', i);
   full_path = strcat(dir_name, '/', image_name);
   % disp(full_path);
   img = imread(full_path);
   img_array_unknown(:,i+1) = reshape(img, [h*w,1]);
end

% Separate training and testing sets
img_male_training = img_array_male(:,1:num_male_training);
img_male_testing = img_array_male(:,(num_male_training+1):num_male_imgs);
img_female_training = img_array_female(:,1:num_female_training);
img_female_testing = img_array_female(:,(num_female_training+1):num_female_imgs);


C = [ img_female_training, img_male_training ];
B = C' * C; % 173x173
[V, D] = eig(B);
%
num_male = size(img_male_training, 2);
num_female = size(img_female_training, 2);
mM = sum(img_male_training, 2)/num_male;
mF = sum(img_female_training, 2)/ num_female;

A = zeros(h*w, num_male + num_female);
for i = 1:(num_male + num_female)
    Ce = C*V(:,i);
    Lambda = D(i,i);
    A(:,i) = Ce * sqrt(Lambda)/norm(Ce,2);
end
% Compute y
yy = A' * (mF - mM);
% Solve for x
z = (D*D*(V')) \ yy;
ww = C*z;
% Show Fisher face
% imshow(reshape(ww, [h,w]),[]);
% Calculate fisher face boundary
% wwb = (ww' * mM + ww' * mF)/2;

score_male_test = zeros(1, size(img_male_testing, 2));
score_female_test = zeros(1, size(img_female_testing, 2));
for i = 1:size(img_male_testing, 2)
    score_male_test(i) = ww' * img_male_testing(:,i);
end

for i = 1:size(img_female_testing, 2)
    score_female_test(i) = ww' * img_female_testing(:,i);
end
figure;
plot(1:10, score_male_test, 'bo', 1:10, score_female_test, 'rx');
title('Values of testing faces projecting onto Fisher face');
xlabel('Test face index');
ylabel('Projection value');
set(gca,'fontsize',18);
legend('Male face projections', 'Female face projection');

%% Part II (6)
lm_male_training = lm_array_male(:,1:num_male_training);
lm_male_testing = lm_array_male(:,(num_male_training+1):num_male_imgs);
lm_female_training = lm_array_female(:,1:num_female_training);
lm_female_testing = lm_array_female(:,(num_female_training+1):num_female_imgs);

% 

C_lm = [lm_female_training, lm_male_training];
B_lm = C_lm' * C_lm;
[V_lm, D_lm] = eig(B_lm);

A_lm = zeros(174, num_male + num_female);
for i = 1:(num_male + num_female)
    Ce = C_lm*V_lm(:,i);
    Lambda = D_lm(i,i);
    A_lm(:,i) = sqrt(Lambda)* Ce /norm(Ce,2) ;
end

mM_lm = zeros(174, 1);
mF_lm = zeros(174, 1);
for i = 1:(num_male)
   mM_lm = mM_lm + lm_male_training(:,i);
end
for i = 1:(num_female)
   mF_lm = mF_lm + lm_female_training(:,i); 
end
mM_lm = mM_lm / size(lm_male_training, 2);
mF_lm = mF_lm / size(lm_female_training, 2);
% mM_lm_2 = sum(lm_male_training, 2)/size(lm_male_training, 2);
% mF_lm_2 = sum(lm_female_training, 2)/size(lm_female_training, 2);
% Compute y
yy_lm = A_lm' * (mF_lm - mM_lm);
% Solve for x
z_lm = (D_lm*D_lm*(V_lm')) \ yy_lm;
ww_lm = C_lm*z_lm;

score_male_lm_test = zeros(1, size(img_male_testing, 2));
score_female_lm_test = zeros(1, size(img_female_testing, 2));

for i = 1:size(img_male_testing, 2)
    score_male_lm_test(i) = ww_lm' * lm_male_testing(:,i);
end

for i = 1:size(img_female_testing, 2)
    score_female_lm_test(i) = ww_lm' * lm_female_testing(:,i);
end

% Plot the scores in 2 dimensional space
figure
scatter(score_male_lm_test, score_male_test, 'b');
hold on;
scatter(score_female_lm_test, score_female_test, 'r');
title('Appearance and key points projection values');
ylabel('Appearance Fisher face projection value');
xlabel('Key point Fisher face projection vlaue');
legend('Male feature', 'Female feature')
set(gca,'fontsize',18);