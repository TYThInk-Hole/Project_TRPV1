using SymbolicRegression
using MLJ
using Plots, DifferentialEquations, Evolutionary, DataDrivenDiffEq, Logging
using XLSX, DataFrames

function runge_kutta4(f, t0, tf, y0, N)
    
    # Initialize arrays
    Currents=ones(1,N+1)*30
    # Currents[Int(40/dt):Int(50/dt)].=10
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

gNa = 120.0
eNa = 115.0
gK = 36.0
eK = -12.0
gCa = 0.4
eCa = 150
gL = 0.3
eL = 10.6

# eNa = 61.5 * log(140 / 10)
# eK = 61.5 * log(5 / 126)
# eCa = 61.5 * log(12 / 10)
# eL = 61.5 * log(151 / 12)
# gNa = 67.15
# gK = 8.91
# gCa = 33.97
# gL = 4.05

tspan = (0,30)
dt = 0.02

VD_Na_Ca = (t, Y, I) -> [((2.5 - 0.1 * (Y[4] + 65)) / (exp(2.5 - 0.1 * (Y[4] + 65)) - 1))*(1-Y[1]) - (4.4 * exp(-(Y[4] + 65) / 18)) * Y[1];
                    (0.6 * exp(-(Y[4] + 65) / 20))*(1-Y[2]) - (1 / (exp(3.0 - 0.1 * (Y[4] + 65)) + 1))*Y[2];
                    ((1 / (1+exp(-(Y[4]+55)/3)))-Y[3])/(17*exp((-(Y[4]+45).^2)/600)+1.5);
                    (gNa * Y[1]^3 * Y[2] * (eNa - (Y[4] + 65)) + gCa * Y[3] * (eCa - (Y[4] + 65)) + gL * (eL - (Y[4] + 65)) + I)]

u0_Na_Ca = [0.5,0.06,0.01, -65]

t, v_naca, I = runge_kutta4(VD_Na_Ca,tspan[1],tspan[2],u0_Na_Ca,Int(tspan[2]/dt))

plot(t,v_naca[4,:])

######################################################################################################
######################################################################################################

VD_Na_K = (t, Y, I) -> [((2.5 - 0.1 * (Y[4] + 65)) / (exp(2.5 - 0.1 * (Y[4] + 65)) - 1))*(1-Y[1]) - (4.4 * exp(-(Y[4] + 65) / 18)) * Y[1];
                    (0.6 * exp(-(Y[4] + 65) / 20))*(1-Y[2]) - (1 / (exp(3.0 - 0.1 * (Y[4] + 65)) + 1))*Y[2];
                    ((0.1 - 0.01 * (Y[4] + 65)) / (exp(1 - 0.1 * (Y[4] + 65)) - 1))*(1-Y[3]) - (0.125 * exp(-(Y[4] + 65) / 80))*Y[3];
                    (gNa * Y[1]^3 * Y[2] * (eNa - (Y[4] + 65)) + gK * Y[3]^4 * (eK - (Y[4] + 65)) + gL * (eL - (Y[4] + 65)) + I)]

u0_Na_K = [0.5,0.06,0.5, -65]

t, v_nak, I = runge_kutta4(VD_Na_K,tspan[1],tspan[2],u0_Na_K,Int(tspan[2]/dt))

plot!(t,v_nak[4,:])

######################################################################################################
######################################################################################################

VD_Na_K_Ca = (t, Y, I) -> [((2.5 - 0.1 * (Y[5] + 0.0)) / (exp(2.5 - 0.1 * (Y[5] + 0.0)) - 1))*(1-Y[1]) - (4.4 * exp(-(Y[5] + 0.0) / 18)) * Y[1];
                    (0.6 * exp(-(Y[5] + 0.0) / 20))*(1-Y[2]) - (1 / (exp(3.0 - 0.1 * (Y[5] + 0.0)) + 1))*Y[2];
                    ((0.1 - 0.01 * (Y[5] + 0.0)) / (exp(1 - 0.1 * (Y[5] + 0.0)) - 1))*(1-Y[3]) - (0.125 * exp(-(Y[5] + 0.0) / 80))*Y[3];
                    ((1 / (1+exp(-(Y[5]+55)/3)))-Y[4])/(17*exp((-(Y[5]+45).^2)/600)+1.5);
                    (gNa * Y[1]^3 * Y[2] * (eNa - (Y[5] + 0.0)) + gK * Y[3]^4 * (eK - (Y[5] + 0.0)) + gCa * Y[4] * (eCa - (Y[5] + 0.0)) + gL * (eL - (Y[5] + 0.0)) + I)]

u0_Na_K_Ca = [0.5,0.06,0.5,0.1, -55]

t, v_nakca, I = runge_kutta4(VD_Na_K_Ca,tspan[1],tspan[2],u0_Na_K_Ca,Int(tspan[2]/dt))

plot(t,v_nakca[5,:])

