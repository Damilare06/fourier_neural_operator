%Number of grid points on [0,1]^2 
%i.e. uniform mesh with step h=1/(s-1)
N = 10000;
s = 256;
% import the a+apd_sub
%delta_data = matfile('a_p_delta_darcy_r256_N100.mat');
delta_data = matfile('/gpfs/u/home/MPFS/MPFSadsj/barn/dev/Externship/fourier_neural_operator/pred/apd_darcy_r256_N100_e05_l2.mat');
apd_sub = delta_data.apd;
% size(apd_sub);

apd_sub = squeeze(apd_sub(:,:,:,1));
sub = round(s / size(apd_sub, 2));

%Create mesh (only needed for plotting)
[X,Y] = meshgrid(0:(1/(s-1)):1);

%Parameters of covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
%Note that we need alpha > d/2 (here d= 2) 
%Laplacian has zero Neumann boundry
%alpha and tau control smoothness; the bigger they are, the smoother the
%function
alpha = 2;
tau = 3;

coeff = zeros(100, s, s);
sol = zeros(100, s, s);
norm_a = GRF2(alpha, tau, s);
% size(norm_a)

a_arr = [repmat(norm_a, 1, 1, N)];
a_arr = permute(a_arr, [3 1 2]);

% solutioning for nearness
for j=1:N 
    %Generate random coefficients from N(0,C)    
    norm_a = GRF2(alpha, tau, s);
    a_arr(j,:,:) = norm_a; 
end

a_sub = a_arr(:,1:sub:end,1:sub:end);

% search for the L2 minimizing input given the subsampling
dist_min = 10000000.000;
idx = 1;
M = size(apd_sub,1);
B_idx = ones(M);
l2_distance = zeros(M, 1);
linf_distance = zeros(M, 1);

for j=1:M
    for k=1:N
        dist = norm(squeeze(apd_sub(j,:,:) - a_sub(k,:,:)));     
        if dist < dist_min
            idx = k;
            dist_min = dist;
        end
    end 
    B_idx(j) = idx;
    l2_distance(j,1) = norm(squeeze(apd_sub(j,:,:) - a_sub(k,:,:)));
    linf_distance(j,1) = norm(squeeze(apd_sub(j,:,:) - a_sub(k,:,:)), 'inf');  
end

disp("Entering darcy loop")

for j=1:M % 100    
    % loop through the built b_j matrix    
    idx = B_idx(j);
    norm_a = squeeze(a_arr(idx,:,:));
    
    %Exponentiate it, so that a(x) > 0
    %Now a ~ Lognormal(0, C)
    %This is done so that the PDE is elliptic
    %lognorm_a = exp(norm_a);
    
    %Another way to achieve ellipticity is to threshhold the coefficients
    thresh_a = zeros(s,s);
    thresh_a(norm_a >= 0) = 12;
    thresh_a(norm_a < 0) = 4;
    
    %Forcing function, f(x) = 1 
    f = ones(s,s);
    
    %Solve PDE: - div(a(x)*grad(p(x))) = f(x)
    %lognorm_p = solve_gwf2(lognorm_a,f);
    thresh_p = solve_gwf2(thresh_a,f);

    coeff(j, :, :) = thresh_a(:,:);
    sol(j, :, :) = thresh_p(:,:);
    disp(j)
end

%Plot coefficients and solutions
%subplot(2,2,1)
%surf(X,Y,lognorm_a); 
%view(2); 
%shading interp;
%colorbar;
%subplot(2,2,2)
%surf(X,Y,lognorm_p); 
%view(2); 
%shading interp;
%colorbar;
%subplot(2,2,3)
%surf(X,Y,thresh_a); 
%view(2); 
%shading interp;
%colorbar;
%subplot(2,2,4)
%surf(X,Y,thresh_p); 
%view(2); 
%shading interp;
%colorbar;


l2_mean_dist = mean(l2_distance, 'all')
linf_mean_dist = mean(linf_distance, 'all')
% save the data for export
save('darcy_r256_N100_e05_l2', 'coeff', 'sol')
