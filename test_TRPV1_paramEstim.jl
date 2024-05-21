using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq,
    Plots, LineSearches
using ModelingToolkit: Interval, infimum, supremum

# Parameters and Variables
@parameters t, gNa, eNa, gK, eK, gL, eL
@variables u₁(..), u₂(..), u₃(..), u₄(..)
Dt = Differential(t)

# Equations
eqs = [Dt(u₁(t)) ~ ((2.5 - 0.1 * (u₄(t) + 55.0)) / (exp(2.5 - 0.1 * (u₄(t) + 55.0)) - 1)) * (1 - u₁(t)) - (4.4 * exp(-(u₄(t) + 55.0) / 18)) * u₁(t),
       Dt(u₂(t)) ~ (0.6 * exp(-(u₄(t) + 55.0) / 20)) * (1 - u₂(t)) - (1 / (exp(3.0 - 0.1 * (u₄(t) + 55.0)) + 1)) * u₂(t),
       Dt(u₃(t)) ~ ((0.1 - 0.01 * (u₄(t) + 55.0)) / (exp(1 - 0.1 * (u₄(t) + 55.0)) - 1)) * (1 - u₃(t)) - (0.125 * exp(-(u₄(t) + 55.0) / 80)) * u₃(t),
       Dt(u₄(t)) ~ (gNa * u₁(t)^3 * u₂(t) * (eNa - (u₄(t) + 55.0)) + gK * u₃(t)^4 * (eK - (u₄(t) + 55.0)) + gL * (eL - (u₄(t) + 55.0)) + 3.0)]

# Initial conditions and domains
bcs = [u₁(0) ~ 0.5, u₂(0) ~ 0.06, u₃(0) ~ 0.5, u₄(0) ~ -55.0]
domains = [t ∈ Interval(0.0, 30.0)]
dt = 0.02

# ODE problem definition
function VDIC!(du, u, p, t)
    du[1] = ((2.5 - 0.1 * (u[4] + 55.0)) / (exp(2.5 - 0.1 * (u[4] + 55.0)) - 1)) * (1 - u[1]) - (4.4 * exp(-(u[4] + 55.0) / 18)) * u[1]
    du[2] = (0.6 * exp(-(u[4] + 55.0) / 20)) * (1 - u[2]) - (1 / (exp(3.0 - 0.1 * (u[4] + 55.0)) + 1)) * u[2]
    du[3] = ((0.1 - 0.01 * (u[4] + 55.0)) / (exp(1 - 0.1 * (u[4] + 55.0)) - 1)) * (1 - u[3]) - (0.125 * exp(-(u[4] + 55.0) / 80)) * u[3]
    du[4] = (p[1] * u[1]^3 * u[2] * (p[2] - (u[4] + 55.0)) + p[3] * u[3]^4 * (p[4] - (u[4] + 55.0)) + p[5] * (p[6] - (u[4] + 55.0)) + 3.0)
end

p = [120.0, 115.0, 36.0, -12.0, 0.4, 10.6]
u0 = [0.5; 0.06; 0.5; -55.0]
tspan = (0.0, 30.0)
prob = ODEProblem(VDIC!, u0, tspan, p)
sol = solve(prob, Tsit5(), tstops = LinRange(0.0, 30.0, 1501))
ts = [infimum(d.domain):0.02:supremum(d.domain) for d in domains][1]

# Function to get data from solution
function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us, ts_]
end

data = getData(sol)
(u_, t_) = data
len = length(data[2])

# Neural network construction
input_ = length(domains)
n = 5
chain1 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))
chain2 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))
chain3 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))
chain4 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))

depvars = [:u₁, :u₂, :u₃, :u₄]

# Additional loss function
function additional_loss(phi, θ, p)
    return sum(sum(abs2, phi[ii](t_, θ[depvars[ii]]) .- u_[[ii], :]) / len for ii in 1:4)
end

discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3, chain4], 
    NeuralPDE.QuadratureTraining(; abstol=1e-6, reltol=1e-6, batch=1501), param_estim=true, 
    additional_loss=additional_loss)

# PDE System definition
@named pde_system = PDESystem(eqs, bcs, domains, [t], [u₁(t), u₂(t), u₃(t), u₄(t)], [gNa, eNa, gK, eK, gL, eL], 
    defaults=Dict(gNa => 10.0, eNa => 10.0, gK => 10.0, eK => 10.0, gL => 10.0, eL => 10.0))

prob = NeuralPDE.discretize(pde_system, discretization)

iter_count = 0
callback = function (p, l)
    global iter_count += 1
    println("Current iteration: $iter_count / 1500, Current loss: $l")
    return false
end

res = Optimization.solve(prob, BFGS(linesearch=BackTracking()); maxiters=1000, callback=callback)
p_ = res.u[(end-5):end]

tt = collect(LinRange(0, 30, 1501))
minimizers = [res.u.depvar[depvars[i]] for i in 1:4]
ts = [infimum(d.domain):(0.02):supremum(d.domain) for d in domains][1]
u_predict = [[discretization.phi[j]([t], minimizers[j])[1] for t in ts] for j in 1:4]
plot(tt,sol[4,:])
plot!(tt,u_predict[4,:][1])

df_p = DataFrame(gNa=p_[1], gK=p_[2], gCa=p_[3], gL=p_[4])
CSV.write("parameters$i.csv", df_p)

df_predict = DataFrame(ts=ts, u_predict=u_predict, t_=t_[1, :], V=V[1, :])
CSV.write("data$i.csv", df_predict)
