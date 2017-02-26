function [ cell_array ] = lmMATToCellarray( landmark_vec )
%lmMATToCellarray Summary of this function goes here
%   Detailed explanation goes here
cell_array = cell(87, 1);
if length(landmark_vec) ~= 174
    error('Size of the landmark vector is wrong!');
    return; 
end
for i = 1:87
   temp = zeros(1,2);
   temp(1,1) = landmark_vec(i*2-1);
   temp(1,2) = landmark_vec(i*2);
   cell_array{i} = temp;
end
end

