% Radom function from N(m, C) on [0 1] where
% C = sigma^2(-Delta + tau^2 I)^(-gamma)
% with periodic, zero dirichlet, and zero neumann boundary.
% Dirichlet only supports m = 0.
% N is the # of Fourier modes, usually, grid size / 2.

% u_0 is generated as a gausian random field, \nu = 0.1
% m = starting point
% mu = mean
% std = sigma


function u = GRF2(N, m, gamma, tau, sigma, type)


if type == "dirichlet"
    m = 0;
end

if type == "periodic"
    my_const = 2*pi;
else
    my_const = pi;
end

my_eigs = sqrt(2)*(abs(sigma).*((my_const.*(1:N)').^2 + tau^2).^(-gamma/2));
% my_eigs = sqrt(2)*(abs(sigma).*((my_const.*(1:N)').^2 + tau^2).^(-gamma/2) + (pert_std.*ones(1,N)'));

if type == "dirichlet"
    alpha = zeros(N,1);
else
    xi_alpha = randn(N,1);
    alpha = my_eigs.*xi_alpha;
end

if type == "neumann"
    beta = zeros(N,1);
else
    xi_beta = randn(N,1);
    beta = my_eigs.*xi_beta;
end

a = alpha/2;
b = -beta/2;

% flip the order of the matrices
c = [flipud(a) - flipud(b).*1i;m + 0*1i;a + b.*1i]; % [8193 X 1]
% pert_gauss = normrnd(0, pert_std, size(c));
% c = c + pert_gauss;

% pert_total = [pert(1:N); 0; pert(N+1:end)];
% size(pert_total)
% add the perturbation to the c vector before generating the chebfun
% c = c + pert_total;

if type == "periodic"
    uu = chebfun(c, [0 1], 'trig', 'coeffs');
%     uu = chebfun(c, [0 1], 'trig');
    % c = [rev(alpha/2) + rev(beta/2)i ] - 4096
    %     [m + 0i                      ] - 1
    %     [alpha/2 - (beta/2)i          ] - 4096
    %  u = uu(t - 0.5)
    u = chebfun(@(t) uu(t - 0.5), [0 1], 'trig');
    %fprintf('uu: %f, u: %f ', range(uu), range(u))
else
    uu = chebfun(c, [-pi pi], 'trig', 'coeffs');
    u = chebfun(@(t) uu(pi*t), [0 1]);
end