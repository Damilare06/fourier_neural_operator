function u = burgers1(init, tspan, s, visc)

% stiff PDE Integrator - exponential integrators 4th-order ETDRK4
S = spinop([0 1], tspan); % generates multiple spin matrices of integer values
dt = tspan(2) - tspan(1); % get the delta_t

S.lin = @(u) + visc*diff(u,2);
S.nonlin = @(u) - 0.5*diff(u.^2);
S.init = init;
u = spin(S,s,dt,'plot','off'); 

