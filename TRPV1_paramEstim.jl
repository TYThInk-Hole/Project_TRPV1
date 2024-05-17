using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq,
    Plots, LineSearches
using ModelingToolkit: Interval, infimum, supremum
using XLSX, DataFrames, CSV

################################# Paramters #################################

eNa = 61.5*log(140/10)
eK = 61.5*log(5/126)
eCa = 61.5*log(2/0.0002)
eL = 61.5*log(149/12)

@parameters t, gNa, gK, gCa, gL
@variables u₁(..), u₂(..), u₃(..), u₄(..), u₅(..)
Dt = Differential(t)
eqs = [Dt(u₁(t)) ~ ((2.5 - 0.1 * (u₅(t) + 0.0)) / (exp(2.5 - 0.1 * (u₅(t) + 0.0)) - 1))*(1-u₁(t)) - (4.4 * exp(-(u₅(t) + 0.0) / 18)) * u₁(t),
    Dt(u₂(t)) ~ (0.6 * exp(-(u₅(t) + 0.0) / 20))*(1-u₂(t)) - (1 / (exp(3.0 - 0.1 * (u₅(t) + 0.0)) + 1))*u₂(t),
    Dt(u₃(t)) ~ ((0.1 - 0.01 * (u₅(t) + 0.0)) / (exp(1 - 0.1 * (u₅(t) + 0.0)) - 1))*(1-u₃(t)) - (0.125 * exp(-(u₅(t) + 0.0) / 80))*u₃(t),
    Dt(u₄(t)) ~ ((1 / (1+exp(-(u₅(t)+55)/3)))-u₄(t))/(17*exp((-(u₅(t)+45).^2)/600)+1.5),
    Dt(u₅(t)) ~ (gNa * u₁(t)^3 * u₂(t) * (eNa - (u₅(t) + 0.0)) + gK * u₃(t)^4 * (eK - (u₅(t) + 0.0)) + gCa * u₄(t) * (eCa - (u₅(t) + 0.0)) + gL * (eL - (u₅(t) + 0.0)) + 20.0)]
bcs = [u₁(0) ~ 0.5, u₂(0) ~ 0.06, u₃(0) ~ 0.5, u₄(0) ~ 0.1, u₅(0) ~ -55.0] #initial values u0
domains = [t ∈ Interval(0.0, 30.0)] #tspan
dt = 0.02 # time Step

df=XLSX.readxlsx("/Users/tyoon/Desktop/Effects of capsaicin treatment on action potential in mouse DRG neurons/CAP_data_separate/CAP 10/003_CAP 10 before/1x Rheobase.xlsx")
data=df[1][:]
V=reshape(data[500:2000,2]*1000,1,:)
tt = reshape(collect(LinRange(0, 30, 1501)),1,:)

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

(u_, t_) = (V, tt)
len = length(t_)

depvars = [:u₁, :u₂, :u₃, :u₄, :u₅]

function additional_loss(phi, θ, p)
    return sum(abs2, phi[5](t_, θ[depvars[5]]) .- u_[[1],:]) / len
end

discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3, chain4, chain5],
    NeuralPDE.QuadratureTraining(; abstol = 1e-6, reltol = 1e-6, batch = 1501), param_estim = true,
    additional_loss = additional_loss)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u₁(t), u₂(t), u₃(t), u₄(t), u₅(t)], [gNa, gK, gCa, gL],
    defaults = Dict([gNa => 80.0, gK => 80.0, gCa => 0.0, gL => -10.0]))

prob = NeuralPDE.discretize(pde_system, discretization)

iter_count = 0
callback = function (p, l)
    global iter_count += 1
    println("Current iteration: $iter_count / 1000, Current loss: $l")
    return false
end
res = Optimization.solve(prob, BFGS(linesearch = BackTracking()); maxiters = 1000, callback = callback)
p_ = res.u[(end - 3):end] # p_ = [9.93, 28.002, 2.667]

minimizers = [res.u.depvar[depvars[5]]]
ts = [infimum(d.domain):(0.02):supremum(d.domain) for d in domains][1]
u_predict = [discretization.phi[5]([t], minimizers[1])[1] for t in ts]
plot(t_[1,:],V[1,:])
plot!(ts, u_predict)

df_p = DataFrame(gNa = p_[1], gK = p_[2], gCa = p_[3], gL = p_[4])
CSV.write("parameters.csv", df_p)

df_predict = DataFrame(ts = ts, u_predict = u_predict, t_ = t_[1,:], V = V[1,:])
CSV.write("data.csv", df_predict)
