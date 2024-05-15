# Load necessary packages
using SymbolicRegression, SymbolicUtils, MLJ
using Plots, DifferentialEquations, Evolutionary, DataDrivenDiffEq, Logging
using XLSX, DataFrames

# Data acquisition
df = XLSX.readxlsx("/Users/tyoon/Desktop/Effects of capsaicin treatment on action potential in mouse DRG neurons/CAP_data_separate/CAP 10/003_CAP 10 before/1x Rheobase.xlsx")
data = df[1][:]
V = data[501:797, 2] * 1000
plot(V)

# Define channel dynamics functions
function alphaM(V)
    return 0.1 * (-V + 25) / (exp((-V + 25) / 10) - 1)
end

function betaM(V)
    return 4 * exp(-V / 18)
end

function alphaH(V)
    return 0.07 * exp(-V / 20)
end

function betaH(V)
    return 1 / (exp((-V + 30) / 10) + 1)
end

function alphaN(V)
    return 0.01 * (-V + 10) / (exp((-V + 10) / 10) - 1)
end

function betaN(V)
    return 0.125 * exp(-V / 80)
end

function alphaE(V)
    return 1 / (1 + exp(-(V + 55) / 3))
end

function betaE(V)
    return 17 * exp((-(V + 45)^2) / 600) + 1.5
end

# Initialize state variables
m, h, n, e = zeros(length(V)), zeros(length(V)), zeros(length(V)), zeros(length(V))
m[1], h[1], n[1], e[1] = 0.3, 0.1, 0.2, 0.4
I, thr, dt = 10, 30, 0.0002

# Constants for the model
gNa, eNa, gCa, eCa, gL, eL = 120, 120, 0.4, 150, 0.3, 10.6

# Simulation loop
for i = 1:length(V)-1
    m[i+1] = m[i] + dt * (alphaM(V[i]) * (1 - m[i]) - betaM(V[i]) * m[i])
    h[i+1] = h[i] + dt * (alphaH(V[i]) * (1 - h[i]) - betaH(V[i]) * h[i])
    n[i+1] = n[i] + dt * (alphaN(V[i]) * (1 - n[i]) - betaN(V[i]) * n[i])
    e[i+1] = e[i] + dt * ((alphaE(V[i]) - e[i]) / betaE(V[i]))
end
k = (m.^3) .* h

# Prepare data for symbolic regression
X1, X2, X3, X4, X5, X6, X7 = hcat(k), hcat(n.^4), hcat(e), hcat(k, n.^4), hcat(n.^4, e), hcat(k, e), hcat(k, n.^4, e)
y = V

# Define the symbolic regression model
model = SRRegressor(
    binary_operators=[+, -, *, /],
    maxsize=50,
    maxdepth=10,
    niterations=100,
    parallelism=:multiprocessing,
    numprocs=8,
    populations=80
)

# Fit models for different combinations
machines = [machine(model, X, y) for X in [X1, X2, X3, X4, X5, X6, X7]]
for mach in machines
    fit!(mach)
end

function calculate_rms_error(observed::Array{Float64}, predictions::Array{Float64})
    # Calculate the squared differences
    squared_errors = (observed .- predictions) .^ 2
    
    # Calculate the mean of the squared errors
    mean_squared_error = mean(squared_errors)
    
    # Return the square root of the mean squared error
    return sqrt(mean_squared_error)
end
# Predictions and error calculation
ypreds = [predict(mach, X) for (mach, X) in zip(machines, [X1, X2, X3, X4, X5, X6, X7])]
rms_errors = [calculate_rms_error(V, ypred) for ypred in ypreds]

# Identify the model with the minimum error
min_error, min_index = findmin(rms_errors)
best_model = machines[min_index]
r = report(best_model)
println("Best best index: ", min_index)
println("Best model equation: ", r.equations[r.best_idx])

# Plot predictions vs. true values
plot(ypreds[min_index][:, 1], label="Predicted", xlabel="Data")
plot!(V, label="target")
