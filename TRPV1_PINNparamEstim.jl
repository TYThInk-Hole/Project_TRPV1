using NeuralPDE, OrdinaryDiffEq
using Lux, Random
using OptimizationOptimJL, LineSearches
using Plots
using XLSX, DataFrames

function Na_K_Ca(u, p, t)
    u₁, u₂, u₃, u₄, u₅ = u
    gNa, gK, gCa, gL = p

    eNa = 61.5 * log(140 / 10)
    eK = 61.5 * log(5 / 126)
    eCa = 61.5 * log(12 / 10)
    eL = 61.5 * log(151 / 12)

    du₁ = ((2.5 - 0.1 * u₅) / (exp(2.5 - 0.1 * u₅) - 1)) * (1 - u₁) - (4.4 * exp(-u₅ / 18)) * u₁
    du₂ = (0.6 * exp(-u₅ / 20)) * (1 - u₂) - (1 / (exp(3.0 - 0.1 * u₅) + 1)) * u₂
    du₃ = ((0.1 - 0.01 * u₅) / (exp(1 - 0.1 * u₅) - 1)) * (1 - u₃) - (0.125 * exp(-u₅ / 80)) * u₃
    du₄ = ((1 / (1 + exp(-u₅ + 55) / 3)) - u₄) / (17 * exp((-(u₅ + 45) .^ 2) / 600) + 1.5)
    du₅ = gNa * u₁^3 * u₂ * (eNa - u₅) + gK * u₃^4 * (eK - u₅) + gCa * u₄ * (eCa - u₅) + gL * (eL - u₅) + 30.0
    [du₁, du₂, du₃, du₄, du₅]
end

tspan = (0.0, 30.0)
u0 = [0.5, 0.06, 0.5, 0.1, -55.0] #initial values u0

prob = ODEProblem(Na_K_Ca, u0, tspan, [120.0, 36.0, 0.4, 0.3])

df = XLSX.readxlsx("/Users/tyoon/Desktop/Effects of capsaicin treatment on action potential in mouse DRG neurons/CAP_data_separate/CAP 10/003_CAP 10 before/1x Rheobase.xlsx")
data = df[1][:]

t_ = reshape(collect(LinRange(0, 30, 1501)), 1, :)
u_ = reshape(data[500:2000, 2] * 1000, 1, :)

rng = Random.default_rng()
Random.seed!(rng, 0)
n = 15
chain = Lux.Chain(
    Lux.Dense(1, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, 4)
)
ps, st = Lux.setup(rng, chain) |> Lux.f64

function additional_loss(phi, θ)
    return sum(abs2, phi(t_, θ) .- u_) / size(u_, 2)
end

opt = LBFGS(linesearch=BackTracking())
alg = NNODE(chain, opt, ps; strategy=WeightedIntervalTraining([0.7, 0.2, 0.1, 0.1], 500),
    param_estim=true, additional_loss=additional_loss)

sol = solve(prob, alg, verbose=true, abstol=1e-8, maxiters=1000, saveat=t_)