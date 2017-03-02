% Extracts edges from CASIA db
clear all;
addpath(genpath('3rdParty'));

%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% set up opts for spDetect (see spDetect.m)
opts = spDetect;
opts.nThreads = 4;  % number of computation threads
opts.k = 512;       % controls scale of superpixels (big k -> big sp)
opts.alpha = .5;    % relative importance of regularity versus data terms
opts.beta = .9;     % relative importance of edge versus color terms
opts.merge = 0;     % set to small value to merge nearby superpixels at end

%% data extraction

source_dir = 'C:\Users\prota\Datasets\CASIA_original\Tp';
dest_dir = 'C:\Users\prota\Datasets\CASIA_original\Tp_borders';

if ~exist(dest_dir)
    mkdir(dest_dir);
    fprintf('Directory created: %s\n', dest_dir);
else
    fprintf('Directory already existing: %s\n', dest_dir);
end

orig_file_list = dir(source_dir);
file_list = orig_file_list(3:end);
t0 = tic;
for i = 780:length(file_list)
    fprintf('%d/%d) Processing file: %s\n', i, length(file_list), file_list(i).name);
    [pathstr,name,ext] = fileparts(fullfile(file_list(i).folder, file_list(i).name));
    I = imread(fullfile(pathstr, file_list(i).name));
    [E,~,~,segs]=edgesDetect(I,model);
    [S,V] = spDetect(I,E,opts); 
%     figure(1); im(I); figure(2); im(V);
    [~,~,U]=spAffinities(S,E,segs,opts.nThreads);
%     figure(3); im(1-U);
    %     figure(1); im(I); figure(2); im(1-E);
    
%     Ib3 = im2bw(U, 0.3);
%     Ib5 = im2bw(U, 0.5);
%     Ib = edge(Ib,'Canny');
    %     figure(3); imshow(Ib);
    imwrite(U, fullfile(dest_dir, [name '_border.png']));
    Ib5 = im2bw(U, 0.5);
    imwrite(Ib5, fullfile(dest_dir, [name '_border_sharp05.png']));
    t1 = toc;
end

fprintf('Processing done in %d seconds\n', t1-t0);

