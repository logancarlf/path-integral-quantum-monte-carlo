using Plots
using Random


function Lagrangian(x0, x1, m, omega, delta_tau) 
    L = 0.5 * m * (((x1 - x0)/delta_tau)^2) + (m * omega^2 * x0^2) 
    return L
end  


function Total_Action(x, m, omega, beta, N, delta_tau)
    S = 0
    for i in 1:N
        S += 0.5*m*Lagrangian(x[i], x[i+1], m, omega, delta_tau)
    return S
    end
end


function Metropolis_Update(x, N, h, id_rate)

    """
    Metropolis_Update performs a single iteration of the Metropolis update
    to converge the markov chain to new positions.

    Inputs:
    x: [Array] Positions of the markov beads before the update
    h: [Float] Size of the new change of the beads
    N: [Int] Number of markov beads

    Returns:
    new_x: [Array] New posiiton of the beads in the markov chain
    h: new h based on the number of accepted position changes
    """

    acceptance_ratio = 0
    random_ints = rand(1:N, N)

    for t in random_ints

        # Create random change in position in (-h,h)
        rand_change = 2 * h * (rand() -0.5)
        new_x = x[t] + rand_change

        # Set periodic boundary conditions
        t_plus = t % N + 1

        # Compute new and old Lagrangian based on change
        S_old = Lagrangian(x[t], x[t_plus], m, omega, delta_tau)
        S_new = Lagrangian(new_x, x[t_plus], m, omega, delta_tau)

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


m = 1
h = 0.5
N = 120
beta = 100
omega = 1
id_rate = 0.8

x = zeros(N)
delta_tau = beta/ N

# Test Lagrangian and Total Action
total_action = Total_Action(x, m, omega, beta, N)
print("Total Action of Harmonic Lagrangian:", total_action, "\n")

# Test Metropolis Metropolis_Update
update = Metropolis_Update(x, N, h, id_rate)
print(update)

print("\n")
