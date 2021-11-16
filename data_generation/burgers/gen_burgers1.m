% number of realizations/ samples to generate
N = 100; % 2048; % 100;

% parameters for the Gaussian random field. TODO: Get the equation for this
gamma = 2.5;
tau = 7;
sigma = 7^(2);

% viscosity
visc = 1/10; % 1/1000;

% grid size
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

% for each of the samples, run the following
for j=1:N
    u0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
    u = burgers1(u0, tspan, s, visc);
    
    u0eval = u0(x);
    a(j,:) = u0eval(1:end-1);
    
    if steps == 1
        output(j,:) = u.values;
    else
        for k=2:(steps+1)
            % shape = (sample_size, time_steps, u_x)
%             figure()
%             u_plot = u{2}.values;%(1:delta_span);
%             plot(u_plot)
%             return
            output(j,k,:) = u{k}.values;
        end
    end
%     if mod(j, 512) == 0s
%         figure()
%         plot(u0), grid on
%     end
    disp(j)
    a_x(j,:) = x(1:end-1);
end

u = output(:,2,:);
u = squeeze(u);
save('burgers_N2048_G8192_inf_2', 'a', 'a_x', 'u')