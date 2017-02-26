%% Project 1 
%% Part I (1) PCA
%% Name:Da Teng
%% NOTE assumeing the face_data directory to be in the same directory
%% of this file
% loading images
curr_dir = pwd;
dir_name = strcat(curr_dir, '/face_data/face');
img_array = [];
start_idx = 0;
num_imgs = 178;
for i = start_idx:(start_idx + num_imgs -1)
   if (i ~= 103)
       image_name = sprintf('face%03d.bmp', i);
       full_path = strcat(dir_name, '/', image_name);
       % disp(full_path);
       img = imread(full_path);
       img_array = cat(3, img_array, img);
   end
end
img_ref = img_array(:,:,size(img_array, 3));
% divide into training and testing set
img_train = img_array(:,:,1:150);
img_test = img_array(:,:,151:177);
clear img_array;
h = size(img_ref, 1);
w = size(img_ref, 2);
NUM_TRAIN_IMAGES = size(img_train, 3);
NUM_TEST_IMAGES = size(img_test, 3);

mean_face = sum(img_train, 3)/NUM_TRAIN_IMAGES;

MAT = zeros(NUM_TRAIN_IMAGES, h * w);
for i = 1:NUM_TRAIN_IMAGES
   current_img = img_train(:,:,i);
   current_img = double(current_img) - mean_face;
   current_img_vec = reshape(current_img, [1, h*w]);
   % imshow(current_img,[]);
   MAT(i, :) = current_img_vec;
   % close;
end

%% Compute eigen value
AA = MAT * MAT';
[V, D] = eig(AA);
figure;
ref_face = reshape(MAT' * V(:,end), [h,w]);
for i = 1:20
    subplot(5,4,i);
    imshow(reshape(MAT' * V(:,end-i+1), [h,w]), []);
end

V_eig = V(:,(end-19):end);

%% Start reconstruction
figure();
for j = 1:NUM_TEST_IMAGES
    cur_reconst = zeros(h, w);
    % cur_mean = mean2(img_test(:,:,j));
    cur_img_test = img_test(:,:,j);
    for i = 0:19
        cur_V = MAT' * V(:,end-i);
        % obtain normalized axises
        cur_V = cur_V / norm(cur_V);
        temp = sum( cur_V .* reshape( double(cur_img_test) - mean_face, [h*w,1])) ...
            * cur_V;
        temp = reshape(temp, [h, w]);
        cur_reconst = cur_reconst + temp;
    end
    cur_reconst = cur_reconst + mean_face;
    subplot(7,8,(j-1)*2+1);
    imshow(img_test(:,:,j), []);
    subplot(7,8,(j-1)*2+2);
    imshow(cur_reconst, []);
end

% Compute the total reconstruction error
error_vec = zeros(40, 1);
for k = 1:40
    for j = 1:NUM_TEST_IMAGES
        cur_reconst = zeros(h, w);
        % cur_mean = mean2(img_test(:,:,j));
        cur_img_test = img_test(:,:,j);
        for i = 0:(k-1)
            cur_V = MAT' * V(:,end-i);
            % obtain normalized axises
            cur_V = cur_V / norm(cur_V);
            temp = sum( cur_V .* reshape( double(cur_img_test) - mean_face, [h*w,1])) ...
                * cur_V;
            temp = reshape(temp, [h, w]);
            cur_reconst = cur_reconst + temp;
        end
        cur_reconst = cur_reconst + mean_face;
        error_vec(k) = error_vec(k) + sum(sum((cur_reconst - double(img_test(:,:,j))).^2))/(256*256);
    end
    error_vec(k) = error_vec(k) / NUM_TEST_IMAGES;
end
% Plotting the error vs. number of eigen faces used.
figure;
plot(error_vec);
xlabel('Number of eigen-faces');
ylabel('error value');
title('Face reconstruction error');


%% PART I (2) Compute landmark of the images
% Loading landmark
% * | * | * | * ... data is stored columnwise
disp('start loading landmark data');
landmark_data = [];
landmark_dir_name = strcat(curr_dir, '/face_data/landmark_87');
for i = start_idx:(start_idx + num_imgs -1)
   if (i ~= 103)
       data_name = sprintf('face%03d_87pt.dat', i);
       full_path = strcat(landmark_dir_name, '/', data_name);
       % disp(full_path);
       cur_landmark_data = importdata(full_path);
       cur_landmark_data = cur_landmark_data(2:end);
       landmark_data = cat(2, landmark_data, cur_landmark_data);
   end
end

% Perform PCA on the landmarking data
clear V;
clear V_eig;
clear D;
NUM_WRAPPING = 10;
landmark_data = landmark_data';
trainLMData = landmark_data(1:150, :);
testLMData = landmark_data(151:177, :); 
M = trainLMData;
% Compute mean wrapping
mean_warpping = sum(M, 1)/size(M, 1);
for i = 1:size(M,1)
    M(i,:) = M(i,:) - mean_warpping;
end
clear landmark_data;
[V, D] = eig(M * M');
% V_eig = V(:,(end-4):end);
V_eig = M' * V;
D_eig = diag(D);
% D_eig = D_eig((end-4):end);

% Plot the first five eigen landmark

for i = 1:5
    figure;
    imshow(ref_face, []);
    hold on;
    cur_lm = V_eig(:,end-i+1)/norm(V_eig(:,end-i+1))*100;
    for j = 1:87
        plot( [ convertPixel(mean_warpping(j*2-1)),...
                convertPixel(cur_lm(j*2-1) + mean_warpping(j*2-1))],...
              [ convertPixel(mean_warpping(j*2)),...
                convertPixel(cur_lm(j*2) + mean_warpping(j*2))],...
            'Color','r','LineWidth',1);
        plot( convertPixel(cur_lm(j*2-1) + mean_warpping(j*2-1)),...
            convertPixel(cur_lm(j*2) + mean_warpping(j*2)), 'b.' )
    end
    str = sprintf('Eigen-warpping %d visualization', i);
    title(str);
end
% Reconstruct the landmark of the test face
% using NUM_WRAPPING number of eigen-wrappings
% from the training data
testLMReconst = zeros(NUM_TEST_IMAGES, 174);
figure;
for i = 1:4%size(testLMData, 1)
    cur_reconst = zeros(174, 1);
    lm_test = testLMData(i,:);
   
    for j = 0:(NUM_WRAPPING-1)
        cur_v = V_eig(:,end-j);
        cur_v = cur_v / norm(cur_v);
        cur_reconst = cur_reconst + sum(cur_v .* (lm_test - mean_warpping)') * cur_v;
    end
    cur_reconst = cur_reconst + mean_warpping';
    testLMReconst(i,:) = cur_reconst';
    %%%%%% Code for ploting landmark reconstruction
    subplot(2,2,i);
    imshow(img_test(:,:,i), []);
    hold on;
    for k = 1:87
        % convert in to integer index
        plot(convertPixel(cur_reconst(k*2-1)), convertPixel(cur_reconst(k*2)), 'r.', 'MarkerSize', 10); 
    end
    for k = 1:87
       plot(lm_test(k*2-1), lm_test(k*2), 'b.', 'MarkerSize', 10); 
    end
    %%%%%%
end
% Calculating errors for landmark reconstruction
error_lm_vec = zeros(1, 40);
for kk = 1:40 % testing for using 1 to 20 eigen-warppings
    for i = 1:size(testLMData, 1)
        cur_reconst = zeros(174, 1);
        lm_test = testLMData(i,:);

        for j = 0:(kk-1)
            cur_v = V_eig(:,end-j);
            cur_v = cur_v / norm(cur_v);
            cur_reconst = cur_reconst + sum(cur_v .* (lm_test - mean_warpping)') * cur_v;
        end
        cur_reconst = cur_reconst + mean_warpping';
        testLMReconst(i,:) = cur_reconst';
        error_lm_vec(kk) = error_lm_vec(kk) + sum(abs(cur_reconst' - lm_test));
        %%%%%%
    end
    error_lm_vec(kk) = error_lm_vec(kk) / size(testLMData, 1);
end
% Plotting reconstruction error ( average distance over the number of 
% eigen-warppings being used
figure;
plot(error_lm_vec);
xlabel('Number of eigen-faces');
ylabel('error value');
title('Landmark reconstruction error');

%% Part I (3)
% Correct landmark for test image is stored in lm_test and the 
% reconstructed landmark is stored in the
img_warpped = zeros(NUM_TRAIN_IMAGES, h*w);
% for i = 1:NUM_TEST_IMAGES
%     img_wrapped(:,:,i) = warpImage_kent(img_test(:,:,i),...
%                         reshape(testLMData(i,:), [2,87])',...
%                         reshape(mean_wrapping, [2,87])');
% end

% Start reconstruction using the wrapped images
for i = 1:NUM_TRAIN_IMAGES
   current_img = img_train(:,:,i);
   current_img = double(current_img) - mean_face;
   % wrap images using landmark data to mean position
   current_img = warpImage_kent(current_img,...
                                reshape(trainLMData(i,:), [2,87])',...
                                reshape(mean_warpping, [2,87])');
   current_img_vec = reshape(current_img, [1, h*w]);
   % imshow(current_img,[]);
   img_warpped(i, :) = current_img_vec;
   % close;
end
clear V;
clear D;
% Compute eigen value
AA = img_warpped * img_warpped';
[V, D] = eig(AA);
figure;
% ref_face2 = reshape(img_wrapped' * V(:,end), [h,w]);
for i = 1:10
    subplot(3,4,i);
    imshow(reshape(img_warpped' * V(:,end-i+1), [h,w]), []);
end

% Start reconstruction
figure;
img_test_warpped = zeros(h, w, NUM_TEST_IMAGES);
for j = 1:NUM_TEST_IMAGES
    cur_reconst = zeros(h, w);
    % warp the image to mean position
    cur_img_test = warpImage_kent(double(img_test(:,:,j)) - mean_face,...
                                reshape(testLMData(j,:), [2,87])',...
                                reshape(mean_warpping, [2,87])');
    img_test_warpped(:,:,j) = cur_img_test;
    % project to eigen faces
    for i = 0:9
        cur_V = img_warpped' * V(:,end-i);
        % obtain normalized axises
        cur_V = cur_V / norm(cur_V);
        temp = sum( cur_V .* reshape( double(cur_img_test), [h*w,1])) ...
            * cur_V;
        temp = reshape(temp, [h, w]);
        cur_reconst = cur_reconst + temp;
    end
    % warp to reconstructed position
    cur_reconst = cur_reconst + mean_face;
    cur_reconst = warpImage_kent(cur_reconst,...
                                reshape(mean_warpping, [2,87])',...
                                reshape(testLMReconst(j,:), [2,87])');
    subplot(7,8,(j-1)*2+1);
    imshow(img_test(:,:,j), []);
    subplot(7,8,(j-1)*2+2);
    imshow(cur_reconst, []);
end
clc;
% Generating reconstruction error
error_both_vec = zeros(1, 40);
for k = 1:40
    for j = 1:NUM_TEST_IMAGES
        cur_reconst = zeros(h, w);
        % warp the image to mean position
        cur_img_test = img_test_warpped(:,:,j);
        % project to eigen faces
        for i = 0:(k-1)
            cur_V = img_warpped' * V(:,end-i);
            % obtain normalized axises
            cur_V = cur_V / norm(cur_V);
            temp = sum( cur_V .* reshape( double(cur_img_test), [h*w,1])) ...
                * cur_V;
            temp = reshape(temp, [h, w]);
            cur_reconst = cur_reconst + temp;
        end
        % warp to reconstructed position
        cur_reconst = warpImage_kent(cur_reconst + mean_face,...
                                    reshape(mean_warpping, [2,87])',...
                                    reshape(testLMReconst(j,:), [2,87])');
        error_both_vec(k) = error_both_vec(k) + sum(sum(abs(double(img_test(:,:,j)) - cur_reconst)))/(256*256);
    %     subplot(7,8,(j-1)*2+1);
    %     imshow(img_test(:,:,j), []);
    %     subplot(7,8,(j-1)*2+2);
    %     imshow(cur_reconst, []);
    end
    error_both_vec(k) = error_both_vec(k) / NUM_TEST_IMAGES;
    str = sprintf('processing k=%d', k);
    disp(str);
end
figure;
plot(error_both_vec);
xlabel('Number of eigen-faces');
ylabel('Error value');
title('Error of reconstruction using both eigen-faces and eigen-warppings');

%% Part I (4)
% Generate random faces by random sampling
NUM_GENERATED_FACES = 20;
NUM_EIGENFACES_USED = 10;
NUM_EIGENWARP_USED = 10;
figure;
for i = 1:NUM_GENERATED_FACES
   % using the eigen-faces in face space at mean position 
   % to generate random faces appearance
   generated_appearance = zeros(h, w);
   for j = 1:NUM_EIGENFACES_USED
      gSample_appearance = normrnd(0, sqrt(D(j,j)));
      v = V(:,end-j+1);
      v = img_warpped' * v;
      v = v / norm(v);
      v = reshape(v, [h,w]);
      generated_appearance = generated_appearance + v * gSample_appearance;
   end
   generated_appearance = generated_appearance + mean_face;
   
   generated_landmark = zeros(174, 1);
   for j = 1:NUM_EIGENWARP_USED
      gSample_warp = normrnd(0, sqrt(D_eig(j)));
      v = V_eig(:,end-j+1);
      v = v / norm(v);
      generated_landmark = generated_landmark + v * gSample_warp;
   end
   generated_landmark = generated_landmark + mean_warpping';
   % Combining appearance and landmark
   generated_face = warpImage_kent(generated_appearance,...
                                    reshape(mean_warpping, [2,87])',...
                                    reshape(generated_landmark, [2,87])');
   subplot(5,4,i);
   imshow(generated_face, []);
end

