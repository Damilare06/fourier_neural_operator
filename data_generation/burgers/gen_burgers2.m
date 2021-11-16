% number of realizations/ samples to generate
N = 100; % 2048; % solve with the same size = 100

delta_data = matfile('a_p_delta_burger_N2048_G8092_inf_2.mat');
delta = delta_data.delta;
% w = warning ('off','all');
delta_sub = delta_data.delta_sub;
ap_delta = delta_data.a_plus_delta;
delta_span = length(ap_delta(1,:));
a_model = delta_data.a;

% parameters for the Gaussian random field. TODO: Get the equation for this
gamma = 2.5;
tau = 7;
sigma = 7^(2);

% viscosity
visc = 1/10; 

% grid size - ABJ 10X the spatial discretization
s = 8192*6; % there is subsampling at every 2**3 gridpoints = 1024
steps = 2; % timesteps

a = zeros(N, s);
if steps == 1
    output = zeros(N, s);
else
    output = zeros(N, steps, s);
    new_output = zeros(N, steps, delta_span);
end
a_x = zeros(N, s);

% discretize the time and space domains
tspan = linspace(0,1,steps+1);
x = linspace(0,1,s+1);


% for each of the samples, run the following
for j=1:N
    % insert the 'a+delta' values as u0, use the subsampled std
    u0 = GRF2(s/2, 0, gamma, tau, sigma, "periodic");

    u = burgers1(u0, tspan, s, visc);
    u0eval = u0(x);
    a(j,:) = u0eval(1:end-1);
    a_eval = a(j,:)';
    
    if steps == 1
        output(j,:) = u.values;
    else
        for k=2:(steps+1)            
            % find the nearest spot to compare apd(j,:)
            for l=1:delta_span
                % get the point
                apd = ap_delta(j, l);  

                idx = dsearchn(a_eval, apd); % return idx of neasest point in a = u0
%                 size(a_eval)
%                 max(a_eval)
%                 a_eval(idx)
                
                figure()
                plot(ap_delta(1,:))
                
%                 figure()
%                 plot(a_model(1,:))
                
                figure()
                plot(a_eval)
                return
            end
            
            % shape = (sample_size, time_steps, u_x)            
            output(j,k,:) = u{k}.values;
        end
    end
    disp(j)
    a_x(j,:) = x(1:end-1);
end

u = output(:,2,:);
u = squeeze(u);
save('burgers_N2048_G8192_gen', 'a', 'a_x', 'u')