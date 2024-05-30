using NeuralPDE, OrdinaryDiffEq
using Lux, Random
using OptimizationOptimJL, LineSearches
using Plots
using LuxCUDA, ComponentArrays
const gpud = gpu_device()
# function lv(u, p, t)
#     u₁, u₂ = u
#     α, β, γ, δ = p
#     du₁ = α * u₁ - β * u₁ * u₂
#     du₂ = δ * u₁ * u₂ - γ * u₂
#     [du₁, du₂]
# end
function rl(u, p, t)
    u₁, u₂, u₃ = u
    α, β, ρ = p
    du₁ = α*(u₂-u₁)
    du₂ = u₁*(ρ-u₃)-u₂
    du₃ = u₁*u₂-β*u₃
    [du₁, du₂, du₃]
end
tspan = (0.0, 5.0)
u0 = [1.0, 0.0, 0.0]
prob = ODEProblem(rl, u0, tspan, [1.0,1.0,1.0])

true_p = [10.0,8/3,28.0]
prob_data = remake(prob, p = true_p)
sol_data = solve(prob_data, Tsit5(), saveat = 0.01)
t_ = sol_data.t
u_ = reduce(hcat, sol_data.u)

rng = Random.default_rng()
Random.seed!(rng, 0)
n = 200
chain = Lux.Chain(
    Lux.Dense(1, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, n, Lux.σ),
    Lux.Dense(n, 3)
)
ps, st = Lux.setup(rng, chain) |> Lux.f64
# ps, st = ps, st |> ComponentArray |> gpud .|> Float64
ps, st = ComponentArray(ps), ComponentArray(st)
ps, st = LuxCUDA.cu(ps), LuxCUDA.cu(st)

function additional_loss(phi, θ)
    return sum(abs2, phi(t_, θ) .- u_) / size(u_, 2)
end

opt = LBFGS(linesearch = BackTracking())
# alg = NeuralPDE.NNODE(chain, opt, ps; strategy = WeightedIntervalTraining([0.7, 0.2, 0.1], 500), param_estim = true, additional_loss = additional_loss)
alg = NeuralPDE.NNODE(chain, opt, ps; strategy = GridTraining(0.01), param_estim = true, additional_loss = additional_loss)

sol = solve(prob, alg, verbose = true, abstol = 1e-6, maxiters = 10000, saveat = t_)
sol.k.u.p
