clear; close all;

ab=imread('mosaicA.bmp');
ground = double(imread("mapA.bmp"));
ab = double(ab)/255;

Ns = 4;
No = 6;
num_classes = 4;

nrows = size(ab,1);
ncols = size(ab,2);

%% Gabor Filter
gabor_channel = cell(Ns, No);
imgA = gaborconvolve(ab,Ns,No,3,2,0.65,1.5);

% Smoothing
for i = 1:Ns
    for j = 1:No
        gabor_channel{i,j} = imgaussfilt(abs(imgA{i,j}), 2, "FilterSize", 3, "Padding", "symmetric");
    end
end

    % Obtain Max and Min feature vectors for each channel
for i = 1:Ns
    for j = 1:No
        max_channel(i,j) = max(max(gabor_channel{i,j}));
        min_channel(i,j) = min(min(gabor_channel{i,j}));
    end
end

% Normalization
for i = 1:Ns
    for j = 1:No
        for k = 1:nrows
            for l = 1:ncols
                gabor_channel{i,j}(k,l) = (gabor_channel{i,j}(k,l) - min_channel(i,j)) / (max_channel(i,j) - min_channel(i,j));
            end
        end
    end
end

% Gabor Outputs Reshaping ... could've done reshape() on entire thing.
gabor_channel_T = [];
for i = 1:Ns
    for j = 1:No
        gabor_channel_T = [gabor_channel_T; reshape(gabor_channel{i,j}, 1, nrows*ncols)];
    end
end

%% K-means Algorithm
iterations = 15;
for i = 1:iterations
    [idx, C, sumd] = kmeans(transpose(gabor_channel_T), num_classes); % 4 textures in mosaic A
    div_sum(:,i) = sumd;
    K{i} = reshape(idx, [256,256]);

    for j = 1:num_classes
        
    end
end

% Add the cluster distances together for a given k-means 
div_sum = sum(div_sum);

[best, best_it] = min(div_sum);
[worst, worst_it] = max(div_sum);


%% display kmeans results
figure
for i = 1:15
    subplot(3,5,i);
    imshow(mat2gray(K{i}));
end

% Accuracy calculations
best_acc = accuracy(ground, K{best_it});
worst_acc = accuracy(ground, K{worst_it});



% EM ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iterations = 20;    % Number of iterations
stop_param = 100;   % Delta Value to stop the iterations


img_bank = cell(4, iterations+1);   % {Image Output; Accuracy; Log-Likelihood; Iterations}
img_bank{1,1} = K{best_it};
img_bank{2,1} = accuracy(ground, K{best_it});
img_bank{3,1} = 0;
img_bank{4,1} = 0;

%% K-means initialization
for i = 1:num_classes
    % Alpha Calculation from K-means Result
    alpha(i) = sum(K{best_it} == i, "all") / (nrows*ncols);

    temp_sigma = [];
    for j = 1:Ns*No
        temp_sigma_col = [];

        % Mu Calculation from K-means Result
        temp_reshape = reshape(gabor_channel_T(j,:), [nrows, ncols]);
        temp_idx = K{best_it} == i;
        temp_idx_mult = temp_idx .* temp_reshape;
        mu(i,j) = sum(sum(temp_idx_mult)) / sum(sum(temp_idx));
        
        % Sigma Calculation from K-means Result
        % Logic to avoid non-matching sizes 
        if sum(temp_idx ~= 0, 'all') ~= sum(temp_idx_mult ~= 0, 'all')
            for k = 1:256
                for l = 1:256
                    if temp_idx(k,l) == 0 && temp_idx_mult(k,l) ~= 0
                        temp_idx_mult(k,l) = 0.0001;
                    end
                    if temp_idx(k,l) ~= 0 && temp_idx_mult(k,l) == 0
                        temp_idx_mult(k,l) = 0.0001;
                    end
                end
            end
        end

        for k = 1:nrows*ncols
            if temp_idx_mult(k) ~= 0
                temp_sigma_col = [temp_sigma_col temp_idx_mult(k)];
            end
        end
        temp_sigma(j,:) = temp_sigma_col;

    end
    mu_sub = repmat(mu(i,:).', 1, sum(temp_idx ~= 0, 'all'));
    temp = temp_sigma - mu_sub;
    sigma(:,:,i) = (temp * temp.') / (sum(temp_idx ~= 0, 'all') - 1);
end


%% Expectation Maximization Algorithm
figure(2)
for i = 1:iterations

    % Expectation Step - Estimate missing class for pixels
    for j = 1:num_classes
        I(:,j) = alpha(j) * mvnpdf(gabor_channel_T.', mu(j,:), sigma(:,:,j));
    end
    I_sum = sum(I, 2);
    I_sum = repmat(I_sum, 1, num_classes);
    I = I ./ I_sum;

    % Maximization Step - Calculate new parameters
    alpha = sum(I,1) / (nrows*ncols);
    divider = repmat(sum(I,1).', 1, Ns*No);
    mu = transpose(gabor_channel_T*I) ./ divider;

    temp_num = [];
    for j = 1:num_classes
        temp_num2 = zeros(Ns*No, Ns*No);
        for k = 1:nrows*ncols
            temp_num = gabor_channel_T(:,k) - mu(j,:).';
            temp_num = temp_num*temp_num.';
            temp_num = temp_num .* I(k,j);
            temp_num2 = temp_num2 + temp_num;
        end
        sum_idx = sum(I,1);
        sigma(:,:,j) = temp_num2 ./ sum_idx(j);
    end
    
    % Log-Likelihood
    for j = 1:num_classes
        lg(:,j) = alpha(j) * mvnpdf(gabor_channel_T.', mu(j,:), sigma(:,:,j));
    end

    % Data Map Classification
    [M, I_plot] = max(I, [], 2);
    img_reshape = reshape(I_plot, [nrows, ncols]);

    % Image Bank
    img_bank{1,i+1} = img_reshape;                  % Image
    img_bank{2,i+1} = accuracy(ground, img_reshape);  % Accuracy
    img_bank{3,i+1} = sum(log(sum(lg,2)));          % Log-Likelihood
    img_bank{4,i+1} = i;                            % Iteration

    imshow(mat2gray(img_bank{1,i+1}))
    % Stop Condition
    if img_bank{3,i+1} - img_bank{3,i} < stop_param
        break
    end
end
final_iteration = i;

%% Video and Plot Construction
v = VideoWriter("Image_Output_for_each_Iteraction.avi");

figure(3);
open(v)
for i = 1:1+final_iteration
    for j = 1:15
        %imshow(mat2gray(img_bank{1, i}));
        N = num2str(i-1);
        string = ['Iteration: ', N];
        string2 = sprintf("Accuracy: %.2f", img_bank{2, i}*100);
        RGB = insertText(mat2gray(img_bank{1, i}),[5 230], string,FontSize=12, TextColor ="black", BoxOpacity = 0.2);
        RGB = insertText(RGB,[150 230], string2, FontSize=12, TextColor ="black", BoxOpacity = 0.2);
        imshow(RGB)
        writeVideo(v, RGB);
    end
end
close(v)

%%
figure(4);
temp1 = [img_bank{2, 1:1+final_iteration}];
temp2 = [img_bank{4, 1:1+final_iteration}];
plot(temp2, temp1)
title("Accuracy vs. Iteration Number Plot");
xlabel("Iteration Number")
ylabel("Accuracy (%)");
xlim([0 final_iteration])

figure(5)
temp1 = [img_bank{3, 2:1+final_iteration}];
temp2 = [img_bank{4, 2:1+final_iteration}];
plot(temp2, temp1)
title("Data Log-Likelihood vs. Iteration Number Plot");
xlabel("Iteration Number")
ylabel("Data Log-Likelihood");
xlim([1 final_iteration])









