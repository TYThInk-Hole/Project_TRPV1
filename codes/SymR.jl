using SymbolicRegression
using MLJ
using Plots, DifferentialEquations, Evolutionary, DataDrivenDiffEq, Logging

function runge_kutta4(f, t0, tf, y0, N)
    t = range(t0, stop=tf, length=N+1)
    y = zeros(length(y0), length(t))
    ydot = zeros(length(y0), length(t))
    y[:, 1] = y0
    ydot[:, 1] .= 0
    
    h = (tf - t0) / N

    for i = 1:N
        k1 = h * f(t[i], y[:, i])
        k2 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k1)
        k3 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k2)
        k4 = h * f(t[i] + h, y[:, i] + k3)

        ydot[:, i+1] = f(t[i], y[:, i])
        y[:, i+1] = y[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    end
    return t, y, ydot
end

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8 / 3

# Lorenz system equations
lorenz = (t, Y) -> [sigma * (Y[2] - Y[1]);
                    Y[1] * (rho - Y[3]) - Y[2];
                    Y[1] * Y[2] - beta * Y[3]]

# Initial conditions
Y0 = [1.0, 1.0, 1.0]  # Julia uses more general floating-point notation

# Time span and steps
t0 = 0.0
tf = 50.0
N = 20000  # High N for a smoother curve

# Solve the ODEs
t, Y, Yd = runge_kutta4(lorenz, t0, tf, Y0, N)

X = transpose(Y[1:3,:])
y = Yd[3,:]

model = SRRegressor(
    binary_operators=[+, -, *, /],
    #unary_operators=[exp],
    niterations=50
)
mach = machine(model, X, y)
fit!(mach)

r = report(mach)

r.equations[r.best_idx]