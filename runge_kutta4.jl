function runge_kutta4(f, t0, tf, y0, N)
    # f - The derivative function of y, f(t, y)
    # t0 - Initial time
    # tf - Final time
    # y0 - Initial condition of y
    # N - Number of steps
    
    # Initialize arrays
    t = LinRange(t0, tf, N+1)
    y = zeros(length(y0), N+1)  # Adjust size if y0 is a vector
    y[:, 1] .= y0
    
    h = (tf - t0) / N  # Step size

    # RK4 Algorithm
    for i in 1:N
        k1 = h * f(t[i], y[:, i])
        k2 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k1)
        k3 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k2)
        k4 = h * f(t[i] + h, y[:, i] + k3)
        
        y[:, i+1] = y[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    end
    
    return t, y
end