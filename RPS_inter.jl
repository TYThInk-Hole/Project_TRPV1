using Random
using SparseArrays

function RPS_inter_individualt(Lsize, reproduction_rate, selection_rate, mobility, para, rn)
    Random.seed!(rn)
    
    # Initialize Lattice
    Lattice = rand(1:4, Lsize, Lsize) .- 1

    # Tracking individual age
    Trace = spzeros(Int, Lsize, Lsize) .+ (Lattice .> 0)
    Trace_time = spzeros(Int, Lsize, Lsize)

    # Mobility
    M = para * 10^(-mobility * (1/10))

    # Exchange rate
    eps = M * (Lsize^2) * 2

    # Define rates
    r1 = reproduction_rate / (reproduction_rate + selection_rate + eps)
    r2 = selection_rate / (reproduction_rate + selection_rate + eps)
    r3 = eps / (reproduction_rate + selection_rate + eps)

    A = [1 0; -1 0; 0 1; 0 -1]

    Flag = true
    generation = 0
    Data_inter = Any[]
    
    while Flag
        Point = true
        G = 1
        generation += 1
        # println("Generation: ", generation)
        stack_inter = zeros(Int, Lsize^2, 3)

        while Point
            p = rand(Lsize^2)

            R = rand(1:Lsize, Lsize^2, 2)
            rr = rand(1:4, Lsize^2)

            Cpre = [R[:,1] .+ A[rr,1] R[:,2] .+ A[rr,2]]
            C1 = Cpre .> Lsize
            C2 = Cpre .< 1

            # Correct boundary conditions
            C = Cpre
            C[C1 .| C2] .= mod1.(Cpre[C1 .| C2], Lsize)

            for i in 1:Lsize^2
                neighbor = Lattice[C[i,1], C[i,2]]
                main = Lattice[R[i,1], R[i,2]]
                neighbor_trace = Trace[C[i,1], C[i,2]]
                main_trace = Trace[R[i,1], R[i,2]]
                neighbor_trace_time = Trace_time[C[i,1], C[i,2]]
                main_trace_time = Trace_time[R[i,1], R[i,2]]

                if p[i] < r1 # reproduction
                    if neighbor == 0 && main != 0
                        G += 1
                        Lattice[C[i,1], C[i,2]] = main
                        Trace[C[i,1], C[i,2]] = 1
                        Trace_time[C[i,1], C[i,2]] = 1
                    elseif neighbor != 0 && main == 0
                        G += 1
                        Lattice[R[i,1], R[i,2]] = neighbor
                        Trace[R[i,1], R[i,2]] = 1
                        Trace_time[R[i,1], R[i,2]] = 1
                    end
                elseif r1 < p[i] < r1 + r2 # selection
                    if neighbor != 0 && main != 0
                        # chain
                        if neighbor - main == 1
                            if Lattice[C[i,1], C[i,2]] == 2
                                stack_inter[G, 2] = Trace[C[i,1], C[i,2]]
                            elseif Lattice[C[i,1], C[i,2]] == 3
                                stack_inter[G, 3] = Trace[C[i,1], C[i,2]]
                            end
                            G += 1
                            Lattice[C[i,1], C[i,2]] = 0
                            Trace[C[i,1], C[i,2]] = 0
                            Trace_time[C[i,1], C[i,2]] = 0
                        elseif neighbor - main == -1
                            if Lattice[C[i,1], C[i,2]] == 1
                                stack_inter[G, 2] = Trace[R[i,1], R[i,2]]
                            elseif Lattice[C[i,1], C[i,2]] == 2
                                stack_inter[G, 3] = Trace[R[i,1], R[i,2]]
                            end
                            G += 1
                            Lattice[R[i,1], R[i,2]] = 0
                            Trace[R[i,1], R[i,2]] = 0
                            Trace_time[R[i,1], R[i,2]] = 0
                        elseif neighbor == 3 && main == 1
                            if Lattice[C[i,1], C[i,2]] == 3
                                stack_inter[G, 1] = Trace[R[i,1], R[i,2]]
                            end
                            G += 1
                            Lattice[R[i,1], R[i,2]] = 0
                            Trace[R[i,1], R[i,2]] = 0
                            Trace_time[R[i,1], R[i,2]] = 0
                        elseif neighbor == 1 && main == 3
                            if Lattice[C[i,1], C[i,2]] == 1
                                stack_inter[G, 1] = Trace[C[i,1], C[i,2]]
                            end
                            G += 1
                            Lattice[C[i,1], C[i,2]] = 0
                            Trace[C[i,1], C[i,2]] = 0
                            Trace_time[C[i,1], C[i,2]] = 0
                        end
                    end
                elseif r1 + r2 < p[i] < r1 + r2 + r3 # move
                    G += 1
                    Lattice[C[i,1], C[i,2]] = main
                    Lattice[R[i,1], R[i,2]] = neighbor
                    Trace[C[i,1], C[i,2]] = main_trace
                    Trace[R[i,1], R[i,2]] = neighbor_trace
                    Trace_time[C[i,1], C[i,2]] = main_trace_time
                    Trace_time[R[i,1], R[i,2]] = neighbor_trace_time
                end
                if G == Lsize^2
                    Point = false
                    break
                end
            end
        end
        Trace = Trace .+ (Trace .!= 0) .- Trace_time

        Trace_A = sparse(reshape(Trace .* (Lattice .== 1), Lsize^2))
        Trace_B = sparse(reshape(Trace .* (Lattice .== 2), Lsize^2))
        Trace_C = sparse(reshape(Trace .* (Lattice .== 3), Lsize^2))

        stack_lattice = zeros(Int, Lsize^2, 3)

        stack_lattice[:,1] = Trace_A
        stack_lattice[:,2] = Trace_B
        stack_lattice[:,3] = Trace_C

        Stacks = sparse(stack_inter)

        push!(Data_inter, (Stacks, stack_lattice))

        Trace_time = spzeros(Int, Lsize, Lsize)

        if generation == 10000
            Flag = false
        end
    end
    # return Data_inter
end

# Example usage
Lsize = 200
reproduction_rate = 1
selection_rate = 1
mobility = 50
para = 1
rn = 1

@time A = RPS_inter_individualt(Lsize, reproduction_rate, selection_rate, mobility, para, rn)