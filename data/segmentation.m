function density_map = segmentation(image_path, position_head_path, k)
%function: generate the density map
%@params:
%img_path: path of input image
%position_head_path: path of ground truth containing the position of person's head
%k: the k nearest neighbor
if nargin < 3
    k = 5;
end
image = imread(image_path);
load(position_head_path);
position_head = image_info{1}.location;
distance_mat = distance(position_head, k);
density_map = zeros(size(image,1), size(image,2));
avg_var_list = compute_avg_var(image, position_head, distance_mat);

for pid = 1:size(position_head,1)
    var = avg_var_list(pid);
    ph = [floor(position_head(pid,2)), floor(position_head(pid,1))];
    circlePixels = circle2d(image, var, [ph(1),ph(2)]);
    density_map(circlePixels') = 1;
end
end

function circlePixels = circle2d(input, radius, center)
%function: generate 2d normal distribution
%@params:
%input: input size
%sigma: [sigmay, sigmax]
%center: [centerx, centery]
    gsize = size(input);
    [X1, X2] = meshgrid(1:gsize(1), 1:gsize(2));
    circlePixels = (X1 - center(1)).^2 ...
    + (X2 - center(2)).^2 <= radius.^2;
    %circlePixels = reshape(circlePixels, gsize(1), gsize(2))';
end

function avg_var_list = compute_avg_var(image,position_head, distance_mat)
%function: caculate the distance matrix
%@params:
%k: the k nearest neighbor
head_num = size(position_head, 1);
var_list = zeros(head_num);
avg_var_list = zeros(head_num);

avg_w = floor(size(image,1)/8.0);
avg_h = floor(size(image,2)/8.0);

for pid = 1:head_num
    var_list(pid) = 0.3 * mean(distance_mat(pid, :));
end


for pid_i = 1:head_num
    pos_i_x_l = floor(position_head(pid_i,2))-avg_w;
    pos_i_y_l = floor(position_head(pid_i,1))-avg_h;
    pos_i_x_r = floor(position_head(pid_i,2))+avg_w;
    pos_i_y_r = floor(position_head(pid_i,1))+avg_h;
    count_temp = 0.0;
    for pid_j = 1:head_num
        pos_j_x = floor(position_head(pid_j,2));
        pos_j_y = floor(position_head(pid_j,1)); 
        
        if pos_j_x< pos_i_x_r && pos_j_x> pos_i_x_l && pos_j_y< pos_i_y_r && pos_j_y> pos_i_y_l
            count_temp = count_temp + 1;
            avg_var_list(pid_i) = avg_var_list(pid_i)+var_list(pid_j);
        end       
    end
    avg_var_list(pid_i) = avg_var_list(pid_i)/count_temp; 
end

end

function distance_matrix = distance(position_head, k)
%function: caculate the distance matrix
%@params:
%k: the k nearest neighbor
head_num = size(position_head, 1);
distance_matrix = zeros(head_num, head_num);
for i = 1:head_num
    for j = 1:head_num
        distance_matrix(i, j) = sum((position_head(i,:)-position_head(j,:)).^2);
    end
end
distance_matrix = sort(distance_matrix, 2);
distance_matrix = sqrt(distance_matrix(:, 1:k));
end
