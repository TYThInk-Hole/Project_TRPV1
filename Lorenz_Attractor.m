% Lorenz system parameters
sigma = 10;
rho = 28;
beta = 8/3;

% Lorenz system equations
lorenz = @(t, Y) [sigma * (Y(2) - Y(1)); ...
                  Y(1) * (rho - Y(3)) - Y(2); ...
                  Y(1) * Y(2) - beta * Y(3)];

% Initial conditions
Y0 = [1; 1; 1]; % Small deviation from the origin

% Time span
t0 = 0;
tf = 50;
N = 10000; % Higher N for a smoother curve

% Solve the ODEs
[t, Y] = runge_Kutta4(lorenz, t0, tf, Y0, N);

function [t, y] = Runge_Kutta4(f, t0, tf, y0, N)
    % f - The derivative function of y, f(t, y)
    % t0 - Initial time
    % tf - Final time
    % y0 - Initial condition of y
    % N - Number of steps

    % Initialize arrays
    t = linspace(t0, tf, N+1);
    y = zeros(1, N+1);
    y(1) = y0;
    
    h = (tf - t0) / N; % Step size

    % RK4 Algorithm
    for i = 1:N
        k1 = h * f(t(i), y(i));
        k2 = h * f(t(i) + 0.5*h, y(i) + 0.5*k1);
        k3 = h * f(t(i) + 0.5*h, y(i) + 0.5*k2);
        k4 = h * f(t(i) + h, y(i) + k3);

        y(i+1) = y(i) + (k1 + 2*k2 + 2*k3 + k4) / 6;
    end
end