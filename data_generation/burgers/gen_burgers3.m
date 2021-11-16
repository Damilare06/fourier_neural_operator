% number of realizations/ samples to generate
N = 10000; % Todo: Change this to 1000

delta_data = matfile('a_p_delta_burger_N2048_G8092_e05.mat');
% w = warning ('off','all');
apd_sub = delta_data.apd_sub;
apd_sub = squeeze(apd_sub(:,:,1));
sub = 8;
% size(apd_sub)
% return
% parameters for the Gaussian random field. TODO: Get the equation for this
gamma = 2.5;
tau = 7;
sigma = 7^(2);

% viscosity
visc = 1/10; 

% grid size - ABJ 10X the spatial discretization
s = 8192; % there is subsampling at every 2**3 gridpoints = 1024
steps = 2; % timesteps

a = zeros(N, s);

if steps == 1
    output = zeros(N, s);
else
    output = zeros(N, steps, s);
end
a_x = zeros(N, s);

% discretize the time and space domains
tspan = linspace(0,1,steps+1);
x = linspace(0,1,s+1);
u0 = GRF2(s/2, 0, gamma, tau, sigma, "periodic");
u0_arr = [repmat(u0,1,N)];

% for each of the samples, run the following
for j=1:N % many realizations ~ 000s
    % insert the 'a+delta' values as u0, use the subsampled std
    u0 = GRF2(s/2, 0, gamma, tau, sigma, "periodic");
    u0_arr(:,j) = u0;
    u0eval = u0(x);
    a(j,:) = u0eval(1:end-1);
end

u0_sub = a(:,1:sub:end);


M = size(apd_sub,1);
B_idx = ones(M);
b_j = zeros(M, size(apd_sub,2));
distance = zeros(M, size(apd_sub,2));
Linf = zeros(M, 1);

% loop and search for L2 distances
for j=1:M
    dist0 = 10000000.000;
    idx = 1;
    for k=1:N
        dist = norm(apd_sub(j,:) - u0_sub(k,:));     
        if dist < dist0
            idx = k;
            dist0 = dist;
        end
    end 
    b_j(j,:) = u0_sub(idx,:);
    distance(j,:) = apd_sub(j,:) - u0_sub(j,:);
        
    Linf(j,1) = norm(apd_sub(j,:) - u0_sub(idx,:), inf);
    B_idx(j) = idx;
end

disp("Entering burgers1 loop")

% the solve step would loop over the b_j matrix
for j=1:M % 100
    % loop through the built b_j matrix
    idx = B_idx(j);
    u0 = u0_arr(:,idx);
    u = burgers1(u0, tspan, s, visc);
    
    if steps == 1
        output(j,:) = u.values;
    else
        for k=2:(steps+1)                       
            % shape = (sample_size, time_steps, u_x)            
            output(j,k,:) = u{k}.values;
        end
    end
    disp(j)
    a_x(j,:) = x(1:end-1);
end

u = output(:,2,:);
u = squeeze(u);
m = mean(distance,'all')
LMean = mean(Linf, 'all')
LCount = sum(Linf>0.05)

save('burgers_N100_G1092_e05', 'a', 'a_x', 'u','b_j', 'Linf')
