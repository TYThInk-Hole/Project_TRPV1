using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq,
    Plots, LineSearches
using ModelingToolkit: Interval, infimum, supremum

@parameters t, gNa, gK, gCa, gL, eNa, eK, eCa, eL
@variables u₁(..), u₂(..), u₃(..), u₄(..), u₅(..)
Dt = Differential(t)
eqs = [Dt(u₁(t)) ~ ((2.5 - 0.1 * (u₅(t) + 65)) / (exp(2.5 - 0.1 * (u₅(t) + 65)) - 1))*(1-u₁(t)) - (4.4 * exp(-(u₅(t) + 65) / 18)) * u₁(t),
    Dt(u₂(t)) ~ (0.6 * exp(-(u₅(t) + 65) / 20))*(1-u₂(t)) - (1 / (exp(3.0 - 0.1 * (u₅(t) + 65)) + 1))*u₂(t),
    Dt(u₃(t)) ~ ((0.1 - 0.01 * (u₅(t) + 65)) / (exp(1 - 0.1 * (u₅(t) + 65)) - 1))*(1-u₃(t)) - (0.125 * exp(-(u₅(t) + 65) / 80))*u₃(t),
    Dt(u₄(t)) ~ ((1 / (1+exp(-(u₅(t)+55)/3)))-u₄(t))/(17*exp((-(u₅(t)+45).^2)/600)+1.5),
    Dt(u₅(t)) ~ (gNa * u₁(t)^3 * u₂(t) * (eNa - (u₅(t) + 65)) + gK * u₃(t)^4 * (eK - (u₅(t) + 65)) + gCa * u₄(t) * (eCa - (u₅(t) + 65)) + gL * (eL - (u₅(t) + 65)) + 0.0)]
bcs = [u₁(0) ~ 0.5, u₂(0) ~ 0.06, u₃(0) ~ 0.5, u₄(0) ~ 0.1, u₅(0) ~ -65.0] #initial values u0
domains = [t ∈ Interval(0.0, 17.0)] #tspan
dt = 0.01 # time Step

# construct neural network
input_ = length(domains)
n = 5
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


function Na_K_Ca!(du, u, p, t)
    du[1] = ((2.5 - 0.1 * (u[5] + 65.0)) / (exp(2.5 - 0.1 * (u[5] + 65.0)) - 1))*(1-u[1]) - (4.4 * exp(-(u[5] + 65.0) / 18)) * u[1]
    du[2] = (0.6 * exp(-(u[5] + 65.0) / 20))*(1-u[2]) - (1 / (exp(3.0 - 0.1 * (u[5] + 65.0)) + 1))*u[2]
    du[3] = ((0.1 - 0.01 * (u[5] + 65.0)) / (exp(1 - 0.1 * (u[5] + 65.0)) - 1))*(1-u[3]) - (0.125 * exp(-(u[5] + 65.0) / 80))*u[3]
    du[4] = ((1 / (1+exp(-(u[5]+55.0)/3.0)))-u[4])/(17*exp((-(u[5]+45.0).^2)/600.0)+1.5)
    du[5] = (p[1] * u[1]^3 * u[2] * (p[2] - (u[5] + 65.0)) + p[3] * u[3]^4 * (p[4] - (u[5] + 65.0)) + p[5] * u[4] * (p[6] - (u[5] + 65.0)) + p[7] * (p[8] - (u[5] + 65.0)) + 0.0)
end

u0 = [0.5,0.06,0.5,0.1, -65.0]
tspan = (0.0, 17.0)
p = [120.0, 115.0, 36.0, -12.0, 0.4, 150.0, 0.3, 10.6]

prob = ODEProblem(Na_K_Ca!, u0, tspan, p)
sol = solve(prob, Tsit5(), tstops=LinRange(0:0.01:17.0))
ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domains][1]

function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us, ts_]
end

data = getData(sol)

(u_, t_) = data
len = length(data[2])

depvars = [:u₁, :u₂, :u₃, :u₄, :u₅]
# function additional_loss(phi, θ, p)
#     return sum(sum(abs2, phi[i](t_, θ[depvars[i]]) .- u_[[i], :]) / len for i in 1:1:3)
# end

function additional_loss(phi, θ, p)
    return sum(sum(abs2, phi[i](t_, θ[depvars[i]]) .- u_[[i], :]) / len for i in 1:1:5)
end

discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3, chain4, chain5],
    NeuralPDE.QuadratureTraining(; abstol = 1e-6, reltol = 1e-6, batch = 1700), param_estim = true,
    additional_loss = additional_loss)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [u₁(t), u₂(t), u₃(t), u₄(t), u₅(t)], [gNa, gK, gCa, gL, eNa, eK, eCa, eL],
    defaults = Dict([gNa => 100.0, gK => 100.0, gCa => 10.0, gL => 1.0, eNa => 0.0, eK => 70.0, eCa => 0.0, eL => 0.0]))

prob = NeuralPDE.discretize(pde_system, discretization)

iter_count = 0
callback = function (p, l)
    global iter_count += 1
    println("Current iteration: $iter_count / 2000, Current loss: $l")
    return false
end
res = Optimization.solve(prob, BFGS(linesearch = BackTracking()); maxiters = 2000, callback = callback)
p_ = res.u[(end - 8):end] # p_ = [9.93, 28.002, 2.667]

minimizers = [res.u.depvar[depvars[i]] for i in 1:5]
ts = [infimum(d.domain):(0.01):supremum(d.domain) for d in domains][1]
u_predict = [[discretization.phi[i]([t], minimizers[i])[1] for t in ts] for i in 1:5]
plot!(sol.t,sol[5,:])
plot(ts, u_predict[5,:])