using Plots
using LinearAlgebra
using Random
using Statistics
using BenchmarkTools
using CircularArrays
using Base.Threads
using StatsBase

#import PyPlot as plt

function Action_QHO(x0, x1, x2, m, omega)
#function Action_QHO(x0, x1, m, omega)

    """
    Action_QHO for the quantum Harmonic oscillator where hbar
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
        The Action_QHO of the bead at x0
    """

    # delta_tau has been removed as working in a unitless dimension, S = integrate(Ldt)
    S = 0.5 * m * ((x1 - x0)^2 + (x2 - x0)^2)/delta_tau + (0.5 * m * omega^2 * x0^2)*delta_tau
    return S
end  


function Thermodynamic_energy(x)
    """
    Calculate the energy of the system

    Inputs:
    x: [Array]
        Positions of the markov beads after final update
    """
    n_beads = N

    K = 0
    V = 0
    
    for i in 1:n_beads

        # K += (0.5 * m * (((x[i+1] - x[i]))^2)) 
        K += ((x[i+1] - x[i])^2)
        V += (0.5 * m * omega^2 * x[i]^2) 
    end
    
    V = V/n_beads # <V>
    K = K/n_beads # <K>
    K = 1/2/delta_tau - 1/(4 * lambda * delta_tau^2) * K
    # H = K + V # Definition of thermodynamic estimator
    H =  K + V
    return H, V, K
end

function Virial_energy(x)
    """
    Calculate the energy of the system from virial theorem

    Inputs:
    x: [Array]
        Positions of the markov beads after final update
    """
    n_beads = N

    K = 0
    V = 0
    COM = 0 # Centre of Mass of the beads

    for pos in 1:n_beads
        COM += x[pos]
    end
    COM /= n_beads
    
    for i in 1:n_beads

        K += (dot((x[i] - COM), x[i])) 
        V += (0.5 * m * omega^2 * x[i]^2) 
    end
    
    V = V/n_beads # <V>
    K = K/n_beads # <K>
    K = 1/2/beta + K * 0.5 * m * omega^2
    H = K + V
  
    return H, V, K
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

    acceptance = 0
    random_ints = rand(1:N, N)
    attempted_change = 0
    for t in random_ints

        # Create random change in position in (-h,h)
        rand_change = 2 * h * (rand() - 0.5)
        new_x = x[t] + rand_change
        attempted_change += 1

        # Compute new and old Action_QHO based on change
        S_old = Action_QHO(x[t], x[t+1], x[t-1], m, omega)
        S_new = Action_QHO(new_x, x[t+1], x[t-1], m, omega)

        # Determine whether the change shoud be accepted
        if rand() < exp(-(S_new-S_old))
            x[t] = new_x
            acceptance += 1
            # acceptance_ratio += 1/N
        end
    end

    # New h, Note errors when acceptance_ratio = 0
    # h = h * acceptance_ratio/id_rate 

    # We only want to record energy after thermalisation, hence setting it to true during the N_path run
    if (record_energy == true)
        virial_energy, VV, KV = Virial_energy(x)
        thermo_energy, VT, KT = Thermodynamic_energy(x)

        return x, h, virial_energy, thermo_energy, VV, KV, VT, KT, acceptance, attempted_change
    end
    
    return x, h, acceptance, attempted_change
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
    Correlation_arr = []
    # Thermalise the paths through Metropolis
    for i in 1:N_thermal
        x, h = Metropolis_Update(x, h, false)
    end

    # Store solutions to reuse
    thermal_x = x
    # thermal_h = h

    # Perforn N_sweep more sweeps on the thermalised path
    @threads for i in 1:N_paths # Using different threads to speed up
    # for i in 1:N_paths
        # println("i = $i on thread $(Threads.threadid())")
        # For each new path, reuse original thermalsied path
        # Store energy after each sweep
        x = thermal_x

        # h = 0.1 # thermal_h
        virial_energy_array_per_step = []
        thermo_energy_array_per_step = []

        #=
        vv_array_per_step = []
        kv_array_per_step = []
        vt_array_per_step = []
        kt_array_per_step = []\
        =#
        acc_arr = []
        att_arr = []

        for j in 1:N_sweep
            if j % 10 == 0
                x, h, virial_energy, thermo_energy, VV, KV, VT, KT, acc, att = Metropolis_Update(x, h, true)
                push!(virial_energy_array_per_step, virial_energy)
                push!(thermo_energy_array_per_step, thermo_energy)
                push!(acc_arr, acc)
                push!(att_arr, att)
                push!(Correlation_arr, Correlation(x, N))
                #=
                push!(vv_array_per_step, VV)
                push!(kv_array_per_step, KV)
                push!(vt_array_per_step, VT)
                push!(kt_array_per_step, KT)
                =#
            
            else
                x, h, acc, att= Metropolis_Update(x, h)
                push!(acc_arr, acc)
                push!(att_arr, att)
            end
            
                #=
                Energy_diagram = scatter(1:length(virial_energy_array_per_step), virial_energy_array_per_step, labels = "virial")
                scatter!(1:length(thermo_energy_array_per_step), thermo_energy_array_per_step, labels = "Thermo")
                # scatter!(1:length(kv_array_per_step), kv_array_per_step, labels = "KV")
                # scatter!(1:length(kt_array_per_step), kt_array_per_step, labels = "KT")
                # scatter!(1:length(vv_array_per_step), vv_array_per_step, labels = "VV")
                
                title!("Energy estimator")
                xlabel!("Step")
                ylabel!("Energy")
                display(Energy_diagram)
                # Record the mean energy of each path respectively, where each experience N_sweep averaging
                =#
        
        end

        push!(H_array_virial, mean(virial_energy_array_per_step))
        push!(H_array_thermo, mean(thermo_energy_array_per_step))
        H_array_virial = virial_energy_array_per_step
        H_array_thermo = thermo_energy_array_per_step
        println("the acceptance ratio is: ", sum(acc_arr)/sum(att_arr))
        # println(length(x))
        # println(H_array_virial)
        push!(thermalised_path_collection, x)

    end
    Corr_avg = mean(Correlation_arr)

    return collect(Iterators.flatten(thermalised_path_collection)), H_array_virial, H_array_thermo, Corr_avg
end

function jackknife_error(observables_array)
    bin_width = Int(floor(0.1*length(observables_array))) #block binwidth
    # println("binwidth is: ", bin_width)
    jk_binwidth = length(observables_array)-bin_width #jack knife binwidth, for complementary bins
    n_bins = cld(length(observables_array),bin_width)

    #getting block estimators

    block_bins = collect(Iterators.partition(observables_array,bin_width)) 
    block_estimators = [sum(bin)/bin_width for bin in block_bins]

    #=
    bins = []
    for i in 1:n_bins
        append!(bins,[sample(observables_array,jk_binwidth,replace=false)])
    end
    =#

    #getting bin_variance

    variance_bins = 0.0
    for k in 1:n_bins
        variance_bins += (block_estimators[k] - mean(observables_array))^2
    end
    variance_bins /= 1/(n_bins*(n_bins-1))
    

    #getting jack knife estimators
    jk_estimators = []

    observables_sum = sum(observables_array)
    for i in 1:n_bins
        jk_estimator = observables_sum - (bin_width * block_estimators[i])
        jk_estimator /= jk_binwidth

        append!(jk_estimators,jk_estimator)
    end

    #getting jack knife variance
    variance_jk = 0.0
        for k in 1:n_bins
            variance_jk += (jk_estimators[k] - mean(observables_array))^2
        end
        variance_jk *= (n_bins-1)/n_bins
    return [variance_bins, variance_jk]
end

function Correlation(pos_arr, n_beads)
    correlation = Vector{Float64}(undef, n_beads)
    for Δτ in 1:n_beads
        for bead_one in 1:n_beads
            bead_two = bead_one + Δτ
            correlation[Δτ] += dot(pos_arr[bead_one], pos_arr[bead_two])
        end
    end		
    return correlation ./ n_beads
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
    - A value for the exectation value of position (expected zero)
    """

    # Produce data from thermalised paths
    path_collection, H_array_virial, H_array_thermo, Corr_arr = Collect_Thermalised_Paths(x, h, 1000, 1, 100000)
    
    # Print virial_energy and thermo_energy for the beta afetr averaging among N_paths
    println("Average virial energy:", mean(H_array_virial))
    println("Average thermo energy:", mean(H_array_thermo))
    
    error_V = jackknife_error(H_array_virial)
    error_T = jackknife_error(H_array_thermo)

    # Print average position
    println("Average Position: ", mean(path_collection))
    # println("Average Energy: ", mean(H_array_virial))

    path_collection = path_collection
    # Plot histogram of position of beads on the path
    hist = histogram(path_collection, bins=50, norm=true, xlim=(-5,5), label="PIMC")

    Δτ_arr = (1:N) * delta_tau

    Corr_graph = scatter(Δτ_arr, Corr_arr, label = "C")
    xlabel!("Δτ")
    ylabel!("G(Δτ)")
    title!("Correlation \n β = $(beta), N = $(N)")
    display(Corr_graph)
    # Plot the analytical solution
    #=
    x_analytical = -3:0.01:3
    psi0 = exp.(-(m_0*omega_0/hbar*x_analytical.^2))/sqrt(pi) * sqrt(m_0*omega_0/(hbar))
    plot!(x_analytical,psi0,label="Analytical",grid=false)
    title!("Probability Distribution of x \n β = $(beta), N = $(N)")
    xlabel!("x")
    ylabel!("frequency")
    display(hist)
    =#

    return mean(H_array_virial), mean(H_array_thermo), sqrt(error_V[2]), sqrt(error_T[2])
end

# Plotting of Energy estimator
T_arr = LinRange(0.1, 1, 10)
#T_arr = [0.005]
Beta_arr = 1 ./ T_arr


# Setting of parameters, note that transforming m and omega into "unitless" concept
h = 1.5 # the maximum size of each random move
hbar = 1 # 1.05e-34
global delta_tau = 0.1
global m = 1
global omega = 1

# energy_recorded for each beta
measured_energy_virial = []
measured_energy_thermo = []
measured_energy_virial_err = []
measured_energy_thermo_err = []

for i in eachindex(Beta_arr)

    println("beta element is ", Beta_arr[i])
    global beta = Beta_arr[i]
    global id_rate = 0.6 #ideal rate
    # global delta_tau = beta/N # should set to be fixed
    global N = floor(Int, beta/delta_tau) #no of beads
    if N < 1
        global N = 1
    end
    # global init_pos = CircularArray(zeros(N))
    global init_pos = CircularArray(2 * (rand(N) .- 0.5))
    global lambda = hbar^2/2/m # prefactor in Ceperly paper
    # Test Against Wavefunction
    global estimate_energy_virial, estimate_energy_thermo, eV, eT = @time Wavefunction_Test(init_pos, h)
    push!(measured_energy_virial, estimate_energy_virial)
    push!(measured_energy_thermo, estimate_energy_thermo)
    push!(measured_energy_virial_err, eV)
    push!(measured_energy_thermo_err, eT)
    
end


# Plotting the Analytical results against PIMC
begin
    gr()
    Avg_E_theoretical = hbar*omega/2 .+ hbar*omega*exp.(-omega*hbar.*Beta_arr)./(1 .-exp.(-hbar*omega.*Beta_arr))
    Analytical_QHO(x) = hbar*omega/2 .+ hbar*omega*exp.(-omega*hbar ./ x)./(1 .-exp.(-hbar*omega ./ x))
    # Energy_plot = scatter(T_arr, Avg_E_theoretical, labels="theory", legend=:topleft)
    Energy_plot = plot(Analytical_QHO, 0.1, 1, legend=:topleft)  
    scatter!(T_arr, measured_energy_virial, yerror = measured_energy_virial_err, labels = "virial est")
    scatter!(T_arr, measured_energy_thermo, yerror = measured_energy_thermo_err, labels = "thermo est" )
    xlabel!("T")
    ylabel!("Energy")
    # xlims!(0.1, 0.6)
    title!("1-d Harmonic Oscillator \n Δτ = $(delta_tau)")
    display(Energy_plot)
    # savefig("./figs/HarmonicEnergyVSTemperature_virial.png")
    # savefig("./figs/HarmonicEnergyVSTemperature_thermo.png")
end


#=
# Test Action_QHO and Total Action
total_action = Total_Action(x, m, omega, beta, N)
print("Testing Action of Harmonic Action_QHO:", "\n")
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

