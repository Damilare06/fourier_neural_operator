% f(t) = tanh(3 sin t) - sin (t +1/2) on[-pi, pi]

% f = chebfun(@(t) tanh(3*sin(t)) - sin( t + 1/2), [-pi, pi], 'trig')
% figure() 
% plot(f), grid on


% This should 