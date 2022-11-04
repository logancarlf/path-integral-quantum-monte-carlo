using Plots
using LinearAlgebra
using Random
using Statistics
using BenchmarkTools
using CircularArrays
using Base.Threads

#import PyPlot as plt

function Lagrangian(x0, x1, x2, m, omega)
#function Lagrangian(x0, x1, m, omega)

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
        Here considered unitless m = m_0 * delta_tau
    omega: [Float]
        Frequency of the harmonic potential.
        Here considered unitless omega = omega_0 * delta_tau
    delta_tau:[Float]
        Distance between beads in imaginary time.
    Returns:
    L: [Float]
        The Lagrangian of the bead at x0
    """

    # delta_tau has been removed as working in a unitless dimension, S = integrate(Ldt)
    S = 0.5 * m * (((x1 - x0))^2 + ((x2 - x0))^2) + (0.5 * m * omega^2 * x0^2) 
    return S
end  

function Primitive_action(x0, x1, m, omega)
    """
    Primitive Action from Ceperly Condensed Helium paper.

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
    # L = 0.5 * m * (((x1 - x0)/delta_tau)^2 + ((x2 - x0)/delta_tau)^2) + (0.5 * m * omega^2 * x0^2) 
    S = 1/2 * log(4*pi*lambda*delta_tau) + (x1 - x0)^2 / 4 / lambda / delta_tau + delta_tau/2 * (1/2 * m * omega^2 * (x1^2 + x0^2))
    return S
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
function Thermodynamic_energy(x)
    """
    Calculate the energy of the system

    Inputs:
    x: [Array]
        Positions of the markov beads after final update
    """
    n_beads = length(x)
    x_circ = CircularArray(x) # Create a circular array to avoid B.C. problems

    K = 0
    V = 0
    
    for i in 1:n_beads

        K += (0.5 * m * (((x_circ[i+1] - x_circ[i]))^2)) 
        V += (0.5 * m * omega^2 * x_circ[i]^2) 
    end
    
    V = V/n_beads # <V>
    K = K/n_beads # <K>
    H = 1/2/delta_tau - (K - V) # Definition of thermodynamic estimator
    return H
end

function Virial_energy(x)
    """
    Calculate the energy of the system from virial theorem

    Inputs:
    x: [Array]
        Positions of the markov beads after final update
    """
    n_beads = length(x)
    x_circ = CircularArray(x) # Create a circular array to avoid B.C. problems

    K = 0
    V = 0
    COM = 0 # Centre of Mass of the beads

    for pos in 1:n_beads
        COM += x_circ[pos]
    end
    COM /= n_beads
    
    for i in 1:n_beads

        K += (0.5 * m * omega^2 * (dot((x_circ[i] - COM), x_circ[i]))) 
        # K += (0.5 * m * omega^2 * (dot(x_circ[i], x_circ[i]))) 
        V += (0.5 * m * omega^2 * x_circ[i]^2) 
    end
    
    V = V/n_beads # <V>
    K = K/n_beads # <K>
    H = 1/2/beta + K + V 
    #println("K is ", K)
    #println("V is ", V)
    #println("H is:", H)
  
    return H
end

function Metropolis_Update(x, h, record_energy = false)

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
        t_minus = mod1(t + N - 1, N)   # periodic boundary conditions
        t_plus = t % N + 1

        # Compute new and old Lagrangian based on change
        S_old = Lagrangian(x[t], x[t_plus], x[t_minus], m, omega)
        S_new = Lagrangian(new_x, x[t_plus], x[t_minus], m, omega)
        # S_old = Lagrangian(x[t], x[t_plus], m, omega)
        # S_new = Lagrangian(new_x, x[t_plus], m, omega)
        #S_old = Primitive_action(x[t], x[t_minus], m, omega) # primitive
        #S_new = Primitive_action(new_x, x[t_minus], m, omega)

        # Determine whether the change shoud be accepted
        if rand() < exp(-(S_new-S_old))
            x[t] = new_x
            acceptance_ratio += 1/N
        end
    end

    # Record potential and kinetic energy when required


    # New h, Note errors when acceptance_ratio = 0
    h = h * acceptance_ratio/id_rate 

    # We only want to record energy after thermalisation, hence setting it to true during the N_path run
    if (record_energy == true)
        virial_energy = Virial_energy(x)
        thermo_energy = Thermodynamic_energy(x)

        return x, h, virial_energy, thermo_energy
    end
    
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
    
    # Array to store paths and energy estimators
    thermalised_path_collection = []
    H_array_virial = []
    H_array_thermo = []
    
    # Thermalise the paths through Metropolis
    for i in 1:N_thermal
        x, h = Metropolis_Update(x, h, false)
    end

    anime_plot = Plots.plot(x, delta_tau .* range(1, N), shape=:circle, labels="beads")
    xlabel!("x")
    ylabel!("imaginary time")
    xlims!(-2, 2)
    display(anime_plot)
    # Store solutions to reuse
    thermal_x = x
    thermal_h = h

    # Perforn N_sweep more sweeps on the thermalised path
    @threads for i in 1:N_paths # Using different threads to speed up
    # for i in 1:N_paths
        # println("i = $i on thread $(Threads.threadid())")
        # For each new path, reuse original thermalsied path
        # Store energy after each sweep
        x = thermal_x
        h = 0.1 # thermal_h
        virial_energy_array_per_step = []
        thermo_energy_array_per_step = []

        for j in 1:N_sweep
            x, h, virial_energy, thermo_energy = Metropolis_Update(x, h, true)
            push!(virial_energy_array_per_step, virial_energy)
            push!(thermo_energy_array_per_step, thermo_energy)
            anime_plot = Plots.plot(x, delta_tau .* range(1, N), shape=:circle, labels="beads")
            xlabel!("x")
            ylabel!("imaginary time")
            xlims!(-2, 2)
            display(anime_plot)

            if j == N_sweep
                # Record the mean energy of each path respectively, where each experience N_sweep averaging
                push!(H_array_virial, mean(virial_energy_array_per_step))
                push!(H_array_thermo, mean(thermo_energy_array_per_step))
            end
        end
        # println(length(x))
        # println(H_array_virial)
        push!(thermalised_path_collection, x)
    end

    return collect(Iterators.flatten(thermalised_path_collection)), H_array_virial, H_array_thermo
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
    path_collection, H_array_virial, H_array_thermo = Collect_Thermalised_Paths(x, h, 100, 1, 100)
    
    # Print virial_energy and thermo_energy for the beta afetr averaging among N_paths
    println("Average virial energy:", mean(H_array_virial))
    println("Average thermo energy:", mean(H_array_thermo))
    
    # Print average position
    println("Average Position: ", mean(path_collection))
    # println("Average Energy: ", mean(H_array_virial))

    # Plot histogram of position of beads on the path
    hist = histogram(path_collection, bins=150, norm=true, xlim=(-3,3), label="PIMC")
    
    # Plot the analytical solution
    x = -3:0.01:3
    psi0 = exp.(-(m*omega/hbar*x.^2))/sqrt(pi) * sqrt(m*omega/hbar)
    plot!(x,psi0,label="Analytical",grid=false)
    title!("Probability Distribution of x \n β = $(beta), N = $(N)")
    xlabel!("x")
    ylabel!("frequency")
    display(hist)

    return mean(H_array_virial), mean(H_array_thermo)
end

# Plotting of Energy estimator
T_arr = LinRange(0.0001, 1, 10)
T_arr = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
T_arr = [0.005]
Beta_arr = 1 ./ T_arr

# Setting of parameters, note that transforming m and omega into "unitless" concept
delta_tau = 1 # should set to be fixed
m_0 = 1
m = m_0 * delta_tau # unitless
omega_0 = 1
omega = omega_0 * delta_tau # unitless
h = 1 # the size of each random move
hbar = 1 # 1.05e-34
lambda = hbar^2/2/m # prefactor in Ceperly paper

# energy_recorded for each beta
measured_energy_virial = []
measured_energy_thermo = []

for i in eachindex(Beta_arr)

    
    println("beta element is ", Beta_arr[i])
    global beta = Beta_arr[i]
    #=
    if beta == 1
        global N = 1
    else
        global N = floor(Int, beta / delta_tau) 
    end
    =#
    global N = floor(Int, beta / delta_tau) 
    global id_rate = 0.6 #ideal rate
    global init_pos = zeros(N)

    # Test Against Wavefunction
    global estimate_energy_virial, estimate_energy_thermo = @time Wavefunction_Test(init_pos, h)
    global anime_plot = Plots.plot()
    push!(measured_energy_virial, estimate_energy_virial)
    push!(measured_energy_thermo, estimate_energy_thermo)
end


# Plotting the Analytical results against PIMC
Avg_E_theoretical = hbar*omega_0/2 .+ hbar*omega_0*exp.(-omega_0*hbar.*Beta_arr)./(1 .-exp.(-hbar*omega_0.*Beta_arr))
Energy_plot = scatter(T_arr, Avg_E_theoretical, labels="theory", legend=:topleft)
scatter!(T_arr, measured_energy_virial, labels = "virial est", shape=:+)
scatter!(T_arr, measured_energy_thermo, labels = "thermo est", shape =:x)
xlabel!("T")
ylabel!("Energy")
title!("Harmonic Oscillator")
display(Energy_plot)


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
# @time Wavefunction_Test(x, h)


# Energy estimator form is of:
# hbar*omega/2*[1+exp(-Beta*hbar*omega)]/[1-exp(-Beta*hbar*omega)]
# = hbar*omega/2 + hbar*omega*exp(-Beta*omega*hbar)/[1-exp(-Beta*hbar*omega)]

