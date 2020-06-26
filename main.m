% % % % % % % % % % % % % % % Load Data % % % % % % % % % % % % % % % 
clear all
load('20NewsHome.mat');                                                    %% Sparse representation

% % % % % % % % % % % % % % % Initialization % % % % % % % % % % % % % % % 
vocab = 15; document = 1000; topic = 12; L=10; iteration = 20;

t = full(fea);                                                             %% Full matrix
[value,index] = sort(sum(t),'desc');                                       %% Words in sorted order (descending)
x = t(1:document,index(1:vocab));                                          %% Consider '#vocab' most commonly used words
y = x'./sum(x'); y = y';                                                   %% Normalize each document vectors with L1 = 1
y(isnan(y)) = 0;                                                           %% set nan to 0 (if any)

% % % % % % % % % % 2nd and 3rd order moments % % % % % % % % % %  
eM3 = zeros(vocab,vocab,vocab);
parfor i = 1:document
  eM3 = eM3+outprod(y(i,:),y(i,:),y(i,:));
end
eM3 = eM3./document;                                                       %% Normalize 3rd order moment
eM2 = y'*y./document;                                                      %% Normalize 2nd order moment

% % % % % % % % % % % % % % % Whitening % % % % % % % % % % % % % % % 
[u,s,v] = svd(eM2,'econ');
eW = u(:,1:topic)*diag(diag(s(1:topic,1:topic)).^(-0.5));                  %% Whitening matrix

% % % % % Whetining/orthogonally decomposable tensor % % % % % 
temp3 = tmprod(eM3,eW',3);
temp2 = tmprod(temp3,eW',2);
eM3d = tmprod(temp2,eW',1);

% % % % % % % % % % % % % % % RTPI % % % % % % % % % % % % % % % 
[M3s, emu, ew] = ePlainPI(eM3d, vocab, eW, topic, L, iteration);
rtp1 = frob(eM3-M3s)/frob(eM3);                                            %% Frobenius norm relative error approximation

% % % % % % % % % % Parameter to control sketch size % % % % % % % % % % 
sketchSize = 100;
clear p6 q6 r6 s6

% % % % % % % % % % % % % % % Uniform % % % % % % % % % % % % % % % 
for k = 1:10
  c = 0;
  for j = 1:document
    if (rand() < 200/document)                                             %% Uniform Sampling
      c = c+1;
      y1(c,:) = y(j,:)/(200/document)^(1/3);                               %% Scale and Sample
    end
  end

  yd = y1*eW;                                                              %% Whitening sampled rows

% % % % % % % % % % 3rd order moment from sampled row % % % % % % % % % % 
  sM3d = zeros(topic,topic,topic);
  parfor i = 1:c
    sM3d = sM3d + outprod(yd(i,:),yd(i,:),yd(i,:));
  end
  sM3d = sM3d./document;                                                   %% Normalize 3rd order moment

% % % % % % % % % % % % % % % RTPI % % % % % % % % % % % % % % % 
  [M3s, smu, sw, nM, diff1, diff2, match] = tensorPI(sM3d, vocab, eW, topic, L, iteration, emu, ew, eM3d);
  p1(k,1) = frob(eM3-M3s)/frob(eM3);                                       %% Frobenius norm relative error approximation
  p1(k,2) = rank(y1);
  q1(k,1) = diff1/topic;                                                   %% average L1 distance between true and estimated distribution
  r1(k,1) = nM;
  s1(k,1) = c;                                                             %% sample size
end

count = 1;

% % % % % % % % % % % % % % % other tyoes % % % % % % % % % % % % % % % 
tempmid = zeros(document,vocab,vocab);
lmid = zeros(document,1);
mid = zeros(vocab);
for j = 1:document
  mid = mid + y(j,:)'*y(j,:);
  tempmid(j,:,:) = mid;
  [um,sm,vm] = svd(mid,'econ');
  lmid(j,1) = (y(j,:)*pinv(mid)*y(j,:)');
end

for k = 1:10
  tot0 = 0; tot1 = 0; tot2 = 0; tot3 = 0;
  mid = 0;
  clear y0 y1 y2 y3
  c0 = 0; c1 = 0; c2 = 0; c3 = 0;
  for j = 1:document
    l = lmid(j,1);

% % % % % % % % % % % % % % % online leverage % % % % % % % % % % % % % % % 
    r(j,1) = (l);                                                          %% Online leverage scores
    tot0 = tot0 + r(j,1);
    p(j,1) = r(j,1)/tot0;
    if (rand() < 200*p(j,1))                                               %% Sample probability
      c0 = c0 + 1;
      y0(c0,:) = y(j,:)./(200*p(j,1))^(1/3);                               %% Scale and Sample
    end

% % % % % % % % % % % % % % % LineFilter % % % % % % % % % % % % % % % 
    r(j,1) = min(1,(j)^(1/2)*(l)^(3/2));                                   %% LineFilter sensitivitiy scores with p=3
    tot3 = tot3 + r(j,1);
    p(j,1) = r(j,1)/tot3;
    if (rand() < 1000*p(j,1))                                            %% Sample probability
      c3 = c3 + 1;
      y3(c3,:) = y(j,:)./(1000*p(j,1))^(1/3);                            %% Scale and Sample
    end
  end
    
    
% % % % % % % % % % % % % % % KernelFilter % % % % % % % % % % % % % % % 
  y4 = y3;
  clear y3
  yu = kr(y4',y4');                                                        %% Kernelization
  yu = yu';
  [ul,sl,vl] = svd(y4,'econ');
  [uu,su,vu] = svd(yu,'econ');
  rankl = rank(y4); ranku = rank(yu);
  c3 = 0;
  for i = 1:length(y4(:,1))
    if (rand() < 0.3*i^(1/4)*norm(uu(i,:),2)^2)                                                      %% Sampling probability
      c3 = c3 + 1;
      y3(c3,:) = y4(i,:)./(0.3*i^(1/4)*norm(uu(i,:),2)^2)^(1/3);     %% Scale and Sample
    end
  end
    
  yd0 = y0*eW;                                                             %% Whitening sampled rows (online leverage)
  yd3 = y3*eW;                                                             %% Whitening sampled rows (LineFilter+KernelFilter)

% % % % % % % % % % 3rd order moment from sampled row % % % % % % % % % %
  sM3d = zeros(topic,topic,topic);
  parfor i = 1:c0
    sM3d = sM3d + outprod(yd0(i,:),yd0(i,:),yd0(i,:));
  end
  sM3d = sM3d./document;                                                   %% Normalize 3rd order moment (Online Leverage)

% % % % % % % % % % % % % % % RTPI % % % % % % % % % % % % % % % 
  [M3s, smu, sw, nM, diff1, diff2, match] = tensorPI(sM3d, vocab, eW, topic, L, iteration, emu, ew, eM3d);

  p3(k,1) = frob(eM3-M3s)/frob(eM3);                                       %% Frobenius norm relative error approximation
  p3(k,2) = rank(y0);
  q3(k,1) = diff1/topic;                                                   %% average L1 distance between true and estimated distribution
  r3(k,1) = nM;
  s3(k,1) = c0;                                                            %% sample size

% % % % % % % % % % 3rd order moment from sampled row % % % % % % % % % %
  sM3d = zeros(topic,topic,topic);
  parfor i = 1:c3
    sM3d = sM3d + outprod(yd3(i,:),yd3(i,:),yd3(i,:))./(document);
  end
  sM3d = sM3d./document;                                                   %% Normalize 3rd order moment (LineFilter+KernelFilter)

% % % % % % % % % % % % % % % RTPI % % % % % % % % % % % % % % % 
  [M3s, smu, sw, nM, diff1, diff2, match] = tensorPI(sM3d, vocab, eW, topic, L, iteration, emu, ew, eM3d);

  p6(count,1) = frob(eM3-M3s)/frob(eM3);                                   %% Frobenius norm relative error approximation
  p6(count,2) = rank(y3);
  q6(count,1) = diff1/topic;                                               %% average L1 distance between true and estimated distribution
  r6(count,1) = nM;
  s6(count,1) = c3;                                                        %% sample size
  count = count + 1;
end

% % % % % % % % % % % % % % % Print output % % % % % % % % % % % % % % %
format short
[median(s1(:,1)), median(s3(:,1)), median(s6(:,1))]

format long
% rtp1
[median(q1(:,1)), median(q3(:,1)), median(q6(:,1))]