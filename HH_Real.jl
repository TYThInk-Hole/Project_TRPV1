using Plots, CSV, DataFrames, XLSX

### real data load
df = XLSX.readxlsx("/Users/tyoon/Desktop/Effects of capsaicin treatment on action potential in mouse DRG neurons/CAP_data_separate/CAP 10/003_CAP 10 before/1x Rheobase.xlsx")
data = df[1][:]
T=data[2:3000,1]*1000
Volt=data[2:3000,2]*1000
### HH model 
function runge_kutta4(f, t0, tf, y0, N, param)
    
    # Initialize arrays
    Currents=ones(1,N+1).*param
    # Currents[Int(10/dt):Int(15/dt)].=param
    t = LinRange(t0, tf, N+1)
    y = zeros(length(y0), N+1)  # Adjust size if y0 is a vector
    y[:, 1] .= y0
    
    h = (tf - t0) / N  # Step size

    # RK4 Algorithm
    for i in 1:N
        I1=Currents[i]
        V1=Volt[i]
        k1 = h * f(t[i], y[:, i], I1, V1)
        k2 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k1, I1, V1)
        k3 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k2, I1, V1)
        k4 = h * f(t[i] + h, y[:, i] + k3, I1, V1)
        
        y[:, i+1] = y[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    end
    
    return t, y, Currents
end

gates = (t, Y, I, V) -> [ ((2.5 - 0.1 * (V + 55.0)) / (exp(2.5 - 0.1 * (V + 55.0)) - 1)) * (1 - Y[1]) - (4.4 * exp(-(V + 55.0) / 18)) * Y[1]
(0.6 * exp(-(V + 55.0) / 20)) * (1 - Y[2]) - (1 / (exp(3.0 - 0.1 * (V + 55.0)) + 1)) * Y[2]
((0.1 - 0.01 * (V + 55.0)) / (exp(1 - 0.1 * (V + 55.0)) - 1)) * (1 - Y[3]) - (0.125 * exp(-(V + 55.0) / 80)) * Y[3]
]

VD_Na_K_Ca = (t, Y, I) -> [((0.1 * (Y[5] + 40)) / (-exp(-0.1 * (Y[5] + 40)) + 1)) * (1 - Y[1]) - (4.4 * exp(-(Y[5] + 65) / 18)) * Y[1];
	            (0.07 * exp(-(Y[5] + 65) / 20)) * (1 - Y[2]) - (1 / (exp(- 0.1 * (Y[5] + 35)) + 1)) * Y[2];
	            ((0.01 * (Y[5] + 55)) / (-exp(- 0.1 * (Y[5] + 55)) + 1)) * (1 - Y[3]) - (0.125 * exp(-(Y[5] + 65) / 80)) * Y[3];
		        ((0.02 * (Y[5] + 30)) / (-exp(- 0.1 * (Y[5] + 30)) + 1)) * (1 - Y[4]) - (0.01 * exp(-(Y[5] + 80) / 20)) * Y[4];
	            (gNa * Y[1]^3 * Y[2] * (eNa - Y[5]) + gK * Y[3]^4 * (eK - Y[5]) + gCa * Y[4]^2 * (eCa - Y[5]) + gL * (eL - Y[5]) + I)]

gNa = 120.0
eNa = 115.0#61.5 * log(140 / 10)
gK = 36
eK = -12#61.5 * log(5 / 126)
gCa = 0.1
eCa = 130.9#61.5 * log(12 / 10)
gL = 0.3
eL = 10.613

dt = T[2]-T[1]
tspan = (0.0,T[end])
u0 = [0.0,0.0,0.0]

t, vv, I = runge_kutta4(gates,tspan[1],tspan[2],u0,Int(round(tspan[2]/dt)),30)