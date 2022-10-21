using Plots
using Random
using Statistics
using BenchmarkTools
using Base.Threads

#import PyPlot as plt

function Lagrangian(x0, x1, x2, m, omega, delta_tau)

    """
    Lagrangian for the quantum Harmonic oscillator where hbar
    has been set to 1.

    Inputs:
    x0: [Float]
        Position of the bead of interest.
    x1: [Float]
        Position of the bead after that of interest in the chain.
    m: [Float]
        Mass of the particle expericing HM.
    omega: [Float]
        Frequency of the harmonic potential.
    delta_tau:[Float]
        Distance between beads in imaginary time.

    Returns:
    L: [Float]
        The Lagrangian of the bead at x0
    """
    L = 0.5 * m * (((x1 - x0)/delta_tau)^2 + ((x2 - x0)/delta_tau)^2) + (0.5 * m * omega^2 * x0^2) 
    return L
end  

#=
function Total_Action(x, m, omega, beta, N, delta_tau)
    S = 0
    for i in 1:N
        S += 0.5*m*Lagrangian(x[i], x[i+1], m, omega, delta_tau)
    return S
    end
end
=#
function Hamiltonian(x)

    """
    Calculate the energy of the system

    Inputs:
    x: [Array]
        Positions of the markov beads after final update
    """
    H = 0
    len = length(x)
    for i in 1:len-1
        H += 3/2/delta_tau - (0.5 * m * (((x[i+1] - x[i])/delta_tau)^2)) + (0.5 * m * omega^2 * x[i+1]^2) 
    end
    H += 3/2/delta_tau - (0.5 * m * (((x[1] - x[len])/delta_tau)^2)) + (0.5 * m * omega^2 * x[1]^2)
    return H/len
end

function Metropolis_Update(x, h)

    """
    Metropolis_Update performs a single iteration of the Metropolis update
    to converge the markov chain to new positions.

    Inputs:
    x: [Array]
        Positions of the markov beads before the update
    h: [Float]
        Size of the new change of the beads
    N: [Int]
        Number of markov beads

    Returns:
    new_x: [Array] 
        New posiiton of the beads in the markov chain
    h: [Float]
        new h based on the number of accepted position changes
    """

    acceptance_ratio = 0
    random_ints = rand(1:N, N)

    for t in random_ints

        # Create random change in position in (-h,h)
        rand_change = 2 * h * (rand() - 0.5)
        new_x = x[t] + rand_change

        # Set periodic boundary conditions
        tmin = mod1(t + N - 1, N)   # periodic boundary conditions
        t_plus = t % N + 1

        # Compute new and old Lagrangian based on change
        S_old = Lagrangian(x[t], x[t_plus], x[tmin], m, omega, delta_tau)
        S_new = Lagrangian(new_x, x[t_plus], x[tmin], m, omega, delta_tau)

        # Determine whether the change shoud be accepted
        if rand() < exp(-(S_new-S_old))
            x[t] = new_x
            acceptance_ratio += 1/N
        end
    end

    # New h, Note errors when acceptance_ratio = 0
    h = h * acceptance_ratio/id_rate 
    
    return x, h
end


function Collect_Thermalised_Paths(x, h, N_thermal, N_paths, N_sweep)

    """
    Produces a flattened array of positions of markov beads produced
    from N_thermal iterations of the Metropolis_Update, followed by 
    N_path number of N_sweep iterations that allows for a variation
    in the distribution

    Inputs:
    x: [Array]
        Initial position of markov beads.
    h: [Float]
        Size of change in Metropolis update.
    N_thermal: [Int] Number of initial iterations of Metropolis to 
        produce a viable thermal path.
    N_paths: [Int]
        Number of paths wanted to collect.
    N_sweep: [Int]
        Number of Metropolis Updates after the initial sweep to be
        ungone by the N_path paths.

    Returns:
    paths: [Array]
        Collection of positions of beads on the thermal path
    
    """
    
    # Array to store paths
    thermalised_path_collection = []
    H_array = []
    
    # Thermalise the paths through Metropolis
    for i in 1:N_thermal
        x, h = Metropolis_Update(x, h)
    end

    # Store solutions to reuse
    thermal_x = x
    thermal_h = h

    # Perforn N_sweep more sweeps on the thermalised path
    Threads.@threads for i in 1:N_paths # Using different threads to speed up
    # for i in 1:N_paths
        # println("i = $i on thread $(Threads.threadid())")
        # For each new path, reuse original thermalsied path
        x = thermal_x
        h = 0.1 # thermal_h

        for j in 1:N_sweep
            x, h = Metropolis_Update(x, h)
        end
        H = Hamiltonian(x)

        push!(H_array, H)
        push!(thermalised_path_collection, x)
    end
    return collect(Iterators.flatten(thermalised_path_collection)), H_array
end


function Wavefunction_Test(x, h)

    """
    Test the position wavefunction for the ground state of the
    quantum harmonic oscillator produced by the PIMC method and
    the analytical solution.

    Inputs:
    x: [Array]
        Initial position of markov beads.
    h: [Float]
        Size of change in Metropolis update.

    Returns:
    - Plot of the simulated wavefunction against the analytical
    - A value for the exectation value of positon (expected zero)
    """

    # Produce data from thermalised paths
    path_collection, H_array = Collect_Thermalised_Paths(x, h, 100, 2000, 12)
    
    # Print H_array
    println("H is: ", H_array)
    
    # Print average position
    print("Average Position: ", mean(path_collection))
    println("Average Energy: ", mean(H_array))

    # Plot histogram of position of beads on the path
    hist = histogram(path_collection, bins=150, norm=true, xlim=(-3,3), label="PIMC")
    
    # Plot the analytical solution
    x = -3:0.01:3
    psi0 = exp.(-(x.^2))/sqrt(pi)
    plot!(x,psi0,label="Analytical",grid=false)

    display(hist)
end


m = 1
h = 0.5
N = 4000
beta = 5000
omega = 1
id_rate = 0.8

x = zeros(N)
delta_tau = beta/ N

#=
# Test Lagrangian and Total Action
total_action = Total_Action(x, m, omega, beta, N)
print("Testing Action of Harmonic Lagrangian:", "\n")
print(total_action, "\n")

# Test Metropolis Metropolis_Update
update = Metropolis_Update(x, N, h, id_rate)
print("\nTesting Metropolis Update:", "\n")
print(update, "\n")

# Test Collecting Thermal Paths
thermal_paths = Collect_Thermalised_Paths(x, h, 100, 3, 12)
print("\nTesting Collecting Thermal Paths:", "\n")
print(thermal_paths, "\n")
=#

# Test Against Wavefunction
@time Wavefunction_Test(x, h)



