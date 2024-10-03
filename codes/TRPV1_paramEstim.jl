using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq,
    Plots, LineSearches
using ModelingToolkit: Interval, infimum, supremum
using XLSX, DataFrames, CSV, Random

################################# Load Data #################################

df = XLSX.readxlsx("/Users/tyoon/Desktop/Effects of capsaicin treatment on action potential in mouse DRG neurons/CAP_data_separate/CAP 10/003_CAP 10 before/1x Rheobase.xlsx")
data = df[1][:]
V = reshape(data[500:3000, 2] * 1000, 1, :)
T = reshape((data[500:3000, 1].-data[500,1]) * 1000, 1, :)

################################# Paramters #################################

eNa = 66.6#61.5 * log(140 / 10)
eK = -89.1#61.5 * log(5 / 126)
eCa = 130.9#61.5 * log(12 / 10)
eL = -56#61.5 * log(151 / 12)

@parameters t, gNa, gK, gCa, gL
@variables u₁(..), u₂(..), u₃(..), u₄(..), u₅(..)
Dt = Differential(t)

eqs = [Dt(u₁(t)) ~ ((0.1 * (u₅(t) + 40)) / (-exp(-0.1 * (u₅(t) + 40)) + 1)) * (1 - u₁(t)) - (4.4 * exp(-(u₅(t) + 65) / 18)) * u₁(t),
    Dt(u₂(t)) ~ (0.07 * exp(-(u₅(t) + 65) / 20)) * (1 - u₂(t)) - (1 / (exp(- 0.1 * (u₅(t) + 35)) + 1)) * u₂(t),
    Dt(u₃(t)) ~ ((0.01 * (u₅(t) + 55)) / (-exp(- 0.1 * (u₅(t) + 55)) + 1)) * (1 - u₃(t)) - (0.125 * exp(-(u₅(t) + 65) / 80)) * u₃(t),
    Dt(u₄(t)) ~ ((0.2 * (u₅(t) + 30)) / (-exp(- 0.1 * (u₅(t) + 30)) + 1)) * (1 - u₄(t)) - (0.8 * exp(-(u₅(t) + 80) / 20)) * u₄(t),
    Dt(u₅(t)) ~ (gNa * u₁(t)^3 * u₂(t) * (eNa - u₅(t)) + gK * u₃(t)^4 * (eK - u₅(t)) + gCa * u₄(t)^2 * (eCa - u₅(t)) + gL * (eL - u₅(t)) + 30.0)]
bcs = [u₁(0) ~ 0.5, u₂(0) ~ 0.06, u₃(0) ~ 0.2, u₄(0) ~ 0.01, u₅(0) ~ -56.0] #initial values u0
domains = [t ∈ Interval(0.0, round(T[end] - T[1], digits=1))] #tspan
dt = 0.2 # time Step



# construct neural network
input_ = length(domains)
n = 10
chain1 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
    Dense(n, 1))
chain2 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
    Dense(n, 1))
chain3 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
    Dense(n, 1))
chain4 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
    Dense(n, 1))
chain5 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
    Dense(n, 1))

(u_, t_) = (V, T)
len = length(t_)

depvars = [:u₁, :u₂, :u₃, :u₄, :u₅]

function additional_loss(phi, θ, p)
    return sum(abs2, phi[5](t_, θ[depvars[5]]) .- u_[[1], :]) / len
end

discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3, chain4, chain5],
    NeuralPDE.QuadratureTraining(; abstol=1e-6, reltol=1e-6, batch=20), param_estim=true,
    additional_loss=additional_loss)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u₁(t), u₂(t), u₃(t), u₄(t), u₅(t)], [gNa, gK, gCa, gL],
    defaults=Dict([gNa => 50.0, gK => 50.0, gCa => 0.0, gL => 0.0]))

prob = NeuralPDE.discretize(pde_system, discretization)

iter_count = 0
callback = function (p, l)
    global iter_count += 1
    println("Current iteration: $iter_count / 1000, Current loss: $l")
    return false
end

res = Optimization.solve(prob, BFGS(linesearch=BackTracking()); maxiters=500, callback=callback)
p_ = res.u[(end-3):end]

minimizers = [res.u.depvar[depvars[5]]]
ts = [infimum(d.domain):(0.2):supremum(d.domain) for d in domains][1]
u_predict = [discretization.phi[5]([t], minimizers[1])[1] for t in ts]
plot(t_[1, :], V[1, :])
plot!(ts, u_predict)

i=1
df_p = DataFrame(gNa=p_[1], gK=p_[2], gCa=p_[3], gL=p_[4])
CSV.write("parameters$i.csv", df_p)

df_predict = DataFrame(ts=ts, u_predict=u_predict, t_=t_[1, :], V=V[1, :])
CSV.write("data$i.csv", df_predict)