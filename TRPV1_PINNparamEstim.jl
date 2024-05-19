using NeuralPDE, OrdinaryDiffEq
using Lux, Random
using OptimizationOptimJL, LineSearches
using Plots

function lv(u, p, t)
    u₁, u₂ = u
    α, β, γ, δ = p
    du₁ = α * u₁ - β * u₁ * u₂
    du₂ = δ * u₁ * u₂ - γ * u₂
    [du₁, du₂]
end

tspan = (0.0, 5.0)
u0 = [5.0, 5.0]
prob = ODEProblem(lv, u0, tspan, [1.0, 1.0, 1.0, 1.0])

true_p = [1.5ㅇ, 1.0, 3.0, 1.0]
prob_data = remake(prob, p=true_p)
sol_data = solve(prob_data, Tsit5(), saveat=0.01)
t_ = sol_data.t
u_ = reduce(hcat, sol_data.u)

rng = Random.default_rng()
Random.seed!(rng, 0)
n = 15
chain = Lux.Chain(
    Lux.Dense(1, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, 2)
)
ps, st = Lux.setup(rng, chain) |> Lux.f64

function additional_loss(phi, θ)
    return sum(abs2, phi(t_, θ) .- u_) / size(u_, 2)
end

opt = LBFGS(linesearch=BackTracking())
alg = NNODE(chain, opt, ps; strategy=WeightedIntervalTraining([0.7, 0.2, 0.1], 500),
    param_estim=true, additional_loss=additional_loss)

sol = solve(prob, alg, verbose=true, abstol=1e-8, maxiters=5000, saveat=t_)