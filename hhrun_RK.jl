using SymbolicRegression
using MLJ
using Plots, DifferentialEquations, Evolutionary, DataDrivenDiffEq, Logging
using XLSX, DataFrames

function runge_kutta4(f, t0, tf, y0, N)
    
    # Initialize arrays
    Currents=ones(1,N+1)*0
    Currents[Int(40/dt):Int(50/dt)].=10
    t = LinRange(t0, tf, N+1)
    y = zeros(length(y0), N+1)  # Adjust size if y0 is a vector
    y[:, 1] .= y0
    
    h = (tf - t0) / N  # Step size

    # RK4 Algorithm
    for i in 1:N
        I1=Currents[i]
        k1 = h * f(t[i], y[:, i], I1)
        k2 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k1, I1)
        k3 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k2, I1)
        k4 = h * f(t[i] + h, y[:, i] + k3, I1)
        
        y[:, i+1] = y[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    end
    
    return t, y, Currents
end

VD_Na_K = (t, Y, I) -> [((2.5 - 0.1 * (Y[4] + 65)) / (exp(2.5 - 0.1 * (Y[4] + 65)) - 1))*(1-Y[1]) - (4.4 * exp(-(Y[4] + 65) / 18)) * Y[1];
                    (0.6 * exp(-(Y[4] + 65) / 20))*(1-Y[2]) - (1 / (exp(3.0 - 0.1 * (Y[4] + 65)) + 1))*Y[2];
                    ((0.1 - 0.01 * (Y[4] + 65)) / (exp(1 - 0.1 * (Y[4] + 65)) - 1))*(1-Y[3]) - (0.125 * exp(-(Y[4] + 65) / 80))*Y[3];
                    (gNa * Y[1]^3 * Y[2] * (eNa - (Y[4] + 65)) + gK * Y[3]^4 * (eK - (Y[4] + 65)) + gL * (eL - (Y[4] + 65)) + I)]



gNa = 120.0
eNa = 115.0
gK = 36.0
eK = -12.0
gL = 0.3
eL = 10.6

tspan = (0,100)
dt = 0.001
u0 = [0.5,0.06,0.5, -65]


t, vv, I = runge_kutta4(VD_Na_K,tspan[1],tspan[2],u0,Int(tspan[2]/dt))

plot(t,vv[4,:])

######################################################################################################
######################################################################################################

V = zeros(length(1:Int(tspan[2]/dt)))
m = zeros(length(1:Int(tspan[2]/dt)))
h = zeros(length(1:Int(tspan[2]/dt)))
n = zeros(length(1:Int(tspan[2]/dt)))

thr = 65
m[1], h[1], n[1], V[1] = u0[1], u0[2], u0[3], u0[4]
for i = 1 : length(V)-1
    V[i+1] = V[i] + dt * (gNa * m[i]^3 * h[i] * (eNa - (V[i] + thr)) + 
                      gK * n[i]^4 * (eK - (V[i]+thr)) +
                      gL * (eL - (V[i] + thr)) + I[i])
    m[i+1] = m[i] + dt * (((2.5 - 0.1 * (V[i] + 65)) / (exp(2.5 - 0.1 * (V[i] + 65)) - 1)) * (1 - m[i]) - (4.4 * exp(-(V[i] + 65) / 18)) * m[i])
    h[i+1] = h[i] + dt * ((0.6 * exp(-(V[i] + 65) / 20)) * (1 - h[i]) - (1 / (exp(3.0 - 0.1 * (V[i] + 65)) + 1)) * h[i])
    n[i+1] = n[i] + dt * ((0.1 - 0.01 * (V[i] + 65)) / (exp(1 - 0.1 * (V[i] + 65)) - 1) * (1 - n[i]) - (0.125 * exp(-(V[i] + 65) / 80)) * n[i])
end

plot!(t[1:end-1],V)
plot!(t[1:end-1],vv[4,2:end]-V)

# ######################################################################################################
# ######################################################################################################

# function hhmodel_runge_kutta4(f1, f2, t0, tf, y0, v0, N)
    
#     # Initialize arrays
#     Currents=ones(1,N+1)*0
#     Currents[Int(40/dt):Int(50/dt)].=10
#     t = LinRange(t0, tf, N+1)
#     y = zeros(length(y0), N+1)
#     v = zeros(length(v0), N+1) 
#     # Adjust size if y0 is a vector
#     y[:, 1] .= y0
#     v[:, 1] .= v0
    
#     h = (tf - t0) / N  # Step size

#     # RK4 Algorithm
#     for i in 1:N
#         I1=Currents[i]
#         k1 = h * f1(t[i], y[:, i], I1, v[i])
#         k2 = h * f1(t[i] + 0.5*h, y[:, i] + 0.5*k1, I1, v[i])
#         k3 = h * f1(t[i] + 0.5*h, y[:, i] + 0.5*k2, I1, v[i])
#         k4 = h * f1(t[i] + h, y[:, i] + k3, I1, v[i])
        
#         y[:, i+1] = y[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6

#         kk1 = h * f2(t[i], y[:, i+1], I1, v[i])
#         kk2 = h * f2(t[i] + 0.5*h, y[:, i+1], I1, v[i] + 0.5*kk1)
#         kk3 = h * f2(t[i] + 0.5*h, y[:, i+1], I1, v[i] + 0.5*kk2)
#         kk4 = h * f2(t[i] + h, y[:, i+1], I1, v[i] + kk3)
        
#         v[i+1] = v[i] + (kk1 + 2*kk2 + 2*kk3 + kk4) / 6        
#     end
    
#     return t, y, v, Currents
# end

# VD_Na_K_1 = (t, Y, I, V) -> [((2.5 - 0.1 * (V[1] + 65)) / (exp(2.5 - 0.1 * (V[1] + 65)) - 1))*(1-Y[1]) - (4.4 * exp(-(V[1] + 65) / 18)) * Y[1];
#                     (0.6 * exp(-(V[1] + 65) / 20))*(1-Y[2]) - (1 / (exp(3.0 - 0.1 * (V[1] + 65)) + 1))*Y[2];
#                     ((0.1 - 0.01 * (V[1] + 65)) / (exp(1 - 0.1 * (V[1] + 65)) - 1))*(1-Y[3]) - (0.125 * exp(-(V[1] + 65) / 80))*Y[3]
#                     ]
# Na_K_V_1 = (t, Y, I, V) -> gNa * Y[1]^3 * Y[2] * (eNa - (V[1] + 65)) + gK * Y[3]^4 * (eK - (V[1] + 65)) + gL * (eL - (V[1] + 65)) + I

# gNa = 120.0
# eNa = 115.0
# gK = 36.0
# eK = -12.0
# gL = 0.3
# eL = 10.6

# tspan = (0,100)
# dt = 0.001
# u0 = [0.5,0.06,0.5]
# v0 = -65


# t, Y, ve, I = hhmodel_runge_kutta4(VD_Na_K_1, Na_K_V_1, tspan[1],tspan[2],u0,v0,Int(tspan[2]/dt))

# plot!(t,ve[1,:])