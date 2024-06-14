using SymbolicRegression
using MLJ
using Plots
using XLSX, DataFrames

function HH_rk4(f, g, t0, tf, y0, v0, N, param)
    
    Currents=ones(1,N+1)
    Currents[Int(50/dt):Int(95/dt)].=param
    t = LinRange(t0, tf, N+1)
    y = zeros(length(y0), N+1)
    v = zeros(length(v0), N+1)  # Adjust size if y0 is a vector
    y[:, 1] .= y0
    v[:, 1] .= v0
        
    h = (tf - t0) / N  # Step size

    # RK4 Algorithm
    for i in 1:N
        I1=Currents[i]
        k1 = h * f(t[i], v[i], y[:, i])
        k2 = h * f(t[i] + 0.5*h, v[i], y[:, i] + 0.5*k1)
        k3 = h * f(t[i] + 0.5*h, v[i], y[:, i] + 0.5*k2)
        k4 = h * f(t[i] + h, v[i], y[:, i] + k3)
        
        X = y[:, i] .+ (k1 + 2*k2 + 2*k3 + k4) / 6
        y[:, i+1] = y[:, i] .+ (k1 + 2*k2 + 2*k3 + k4) / 6

        l1 = h * g(t[i], v[i], X, I1)
        l2 = h * g(t[i] + 0.5*h, v[i] + 0.5*l1, X, I1)
        l3 = h * g(t[i] + 0.5*h, v[i] + 0.5*l2, X, I1)
        l4 = h * g(t[i] + h, v[i] + l3, X, I1)

        v[:, i+1] = v[:, i] .+ (l1 + 2*l2 + 2*l3 + l4) / 6
    end
    
    return t, y, v, Currents
end

P_gate = (t, V, Y) -> [((2.5-0.1*(V+56)) ./ (exp(2.5-0.1*(V+56)) -1))*(1-Y[1]) - (4*exp(-(V+56)/18)) * Y[1];
                    (0.07 * exp(-((V+56) / 20)))*(1-Y[2]) - (1/(exp(3.0-0.1*(V+56))+1))*Y[2];
                    ((0.1-0.01*(V+56)) ./ (exp(1-0.1*(V+56)) -1))*(1-Y[3]) - (0.125*exp(-(V+56)/80))*Y[3];
                    ]
function dtVoltage(t, V, X, I)
    dtdv = (gNa * X[1]^3 * X[2] * (eNa - (V+56)) + gK * X[3]^4 * (eK - (V+56)) + gL * (eL - (V+56)) + I)
    return dtdv
end

gNa = 120.0
eNa = 115.0
gK = 36.0
eK = -12.0
gL = 0.3
eL = 10.2

tspan = (0,100)
dt = 0.001
u0 = [0.0, 0.0, 0.0]
v0 = 0

t, m, v, I = HH_rk4(P_gate,dtVoltage,tspan[1],tspan[2],u0,v0,Int(tspan[2]/dt),6)

plot!(v[1,:])

plot((m[1,:].^3).*m[2,:])
plot!((m[3,:].^4))