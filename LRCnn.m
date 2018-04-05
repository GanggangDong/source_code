function LRCnn(tr_dat, gnd_tr, tt_dat, gnd_tt)
% If you use this code, please cite the reference paper. Thank you.
% Ganggang Dong*, Gangyao Kuang, Linjun Zhao, Jun Lu, Min Lu. Joint Nonnegative and Local Linear Regression for Classification in SAR Imagery. 2014 IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2014, 1702-1705, Quebec, Canada.


dFct = 4;

num_tt = length(gnd_tt);
num_tr = length(gnd_tr);

im_h = 80;
im_w = 80;

D = reshape(tr_dat,  [im_h,im_w,num_tr]);
D = imresize(D, [floor(im_h/dFct), floor(im_w/dFct)]);
D = double(reshape(D, [floor(im_h/dFct)*floor(im_w/dFct), num_tr]));

Y = reshape(tt_dat,  [im_h,im_w,num_tt]);
Y = imresize(Y, [floor(im_h/dFct), floor(im_w/dFct)]);
Y = double(reshape(Y, [floor(im_h/dFct)*floor(im_w/dFct), num_tt]));

idx_class = unique(gnd_tr);
num_class = length(idx_class);

kn = 50;
kf = 50;

reCoe0 = zeros(num_tt,num_class);
reCoe1 = reCoe0;
tic
for n = 149:num_tt
    y = Y(:,n);
    for m = 1:num_class
        Dc = D(:,gnd_tr==idx_class(m));
        a = sum(Dc'.*Dc',2);
        b = sum(y'.*y',2);
        distE = bsxfun(@plus,a,b') - 2*Dc'*y; % Euclid
        [~,idx] = sort(distE,'ascend');
        %k = ceil(0.3*size(Dc,2));
        A = Dc(:,idx(1:kn));
        %A = Dc(:,[idx(1:kn);idx(end-kf+1:end)]);
        %x = (subD'*subD)\subD'*y;
        
        x0 = linsolve(A,y);
        reCoe0(n,m) = (norm(y - A*x0));
    end
end
toc
[~,idx] = min(reCoe0,[],2);
fprintf('accuracy = %f\n',sum(idx==gnd_tt)/length(gnd_tt));

% [~,idx] = min(reCoe1,[],2);
% fprintf('accuracy = %f\n',sum(idx==gnd_tt)/length(gnd_tt));
end