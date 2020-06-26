function [M3s, sA_est, slambda, nM, diff1, diff2, match] = tensorPI(sM3d, vocab, eW, topic, L, iteration, mu, w, M3dp)
%  tensorPI: Runs robust tensor power iteration and estimates the latent
%  variables
%   sM3d = Orthogonally decomposable tensor
%   vocab = Dimension of input vector
%   eW = Whitening matrix
%   topic = Number of the latent variables
%   L = parameter for robust tensor power iteration
%   iteration = Parameter for robust tensor power iteration
%   mu = True latent factors
%   w = True latent weights
%   M3dp = True whitened tensor
%
%   M3s = estimated 3rd order moment
%   sA_est = Estimated latent factors
%   slambda = estimated latent weights
%   nM = L1 distance of the worst pair in the matching between true and
%   estimated latent factors.
%   diff1 = sum of l1 norm of the difference vector between true and the
%   estiamted latent factors.
%   diff2 = sum of the difference between true and the estiamted latent 
%   weights
%   match = Index of matched true and estimated latent factors.
% 
for i = 1:topic
    % L random vectors
    sV_old = rand(topic,L);
    for j = 1:L
        % normalize vector before running power iteration
        sV_old(:,j) = sV_old(:,j)./norm(sV_old(:,j),2);
        t = sV_old(:,j);
        % Run power iteration
        for k = 1:iteration
            temp3 = tmprod(sM3d,t',3);
            temp2 = tmprod(temp3,t',2);
            % normalize the vector
            t = temp2./norm(temp2,2);
            temp1 = tmprod(sM3d,t',1);
            temp2 = tmprod(temp1,t',2);
            temp3 = tmprod(temp2,t',3);
            if frob(sM3d - temp3*outprod(t,t,t)) < frob(sM3d)
                spickV(:,j) = t;
                spickLambda(j) = temp3;
            end
        end
    end
    % Pick the best vector
    [value, index] = max(spickLambda);
    t = spickV(:,index);
    % Run power iteration
    for k = 1:iteration
        temp3 = tmprod(sM3d,t',3);
        temp2 = tmprod(temp3,t',2);
        t = temp2./norm(temp2,2);
        temp1 = tmprod(sM3d,t',1);
        temp2 = tmprod(temp1,t',2);
        temp3 = tmprod(temp2,t',3);
        if frob(sM3d - temp3*outprod(t,t,t)) < frob(sM3d)
            sV_est(:,i) = t;
            slambda_est(i) = temp3;
        end
    end
    % Deflate the actual sM3d tensor from with current eVector and eValue
    sM3d = sM3d - slambda_est(i)*outprod(sV_est(:,i),sV_est(:,i),sV_est(:,i));
end
T = zeros(topic,topic,topic);
for i = 1:topic
    T = T + slambda_est(i)*outprod(sV_est(:,i),sV_est(:,i),sV_est(:,i));
end

% Unwhiten/inverse transformation of the estimated eVectors
sA_est = pinv(eW')*(sV_est*diag(slambda_est));

% Correct signs based on the 2 norm of each vector
for i = 1:topic
    checkSign = sA_est(:,i);
    countP = 0;
    countN = 0;
    for j = 1:vocab
        if checkSign(j,1) < 0
            countN = countN + (checkSign(j,1))^2;
        else
            countP = countP + (checkSign(j,1))^2;
        end
    end
    if countN > countP
        checkSign = checkSign.*(-1);
    end
    for j = 1:vocab
        if sign(checkSign(j,1)) < 0
            checkSign(j,1) = checkSign(j,1)*(0);
        end
    end
    checkSign = checkSign./sum(checkSign);
    sA_est(:,i) = checkSign;
end
% % eValue to recoved w
slambda = diag(diag(slambda_est)^(-2));
% normalize w
slambda = slambda./sum(slambda);

% Estimated M3s from recovered mu and recovered w
M3s = zeros(vocab,vocab,vocab);
for i = 1:topic
    M3s = M3s + slambda(i)*outprod(sA_est(:,i),sA_est(:,i),sA_est(:,i));
end

for i = 1:topic
    for j = 1:topic
        match1(i,j) = norm(mu(:,i)-sA_est(:,j),1);
        match2(i,j) = abs(w(i)-slambda(j));
        match3(i,j) = min(norm(mu(:,i)-sA_est(:,j),1)^1,abs(w(i)-slambda(j)));
    end
end
diff1 = 0; diff2 = 0;
for i = 1:topic
    [a1,b1] = min(match3);
    [a2,b2] = min(a1);
    match(i,:) = [b1(b2),b2];
    newMetric(i) = match1(b1(b2),b2);
    diff1 = diff1 + match1(b1(b2),b2);
    diff2 = diff2 + match2(b1(b2),b2);
    match1(b1(b2),:) = realmax*ones(1,topic);
    match1(:,b2) = realmax*ones(topic,1);
    match2(b1(b2),:) = realmax*ones(1,topic);
    match2(:,b2) = realmax*ones(topic,1);
    match3(b1(b2),:) = realmax*ones(1,topic);
    match3(:,b2) = realmax*ones(topic,1);

end
nM = max(newMetric);
nM = frob(T-M3dp)/frob(M3dp);