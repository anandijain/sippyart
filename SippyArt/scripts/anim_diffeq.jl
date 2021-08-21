using DifferentialEquations
using Plots
using FixedPointNumbers

function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

p = [10.0,28.0,8 / 3]
u0 = [1.0;0.0;0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob)

plot(sol, vars=(1, 2, 3); background_color=:transparent)

p = plot(sol, vars=(1, 2, 3); background_color=:transparent, border=:none, axis=nothing, legend=nothing)

using Colors
arr = Array(sol)

pixs = map(x -> RGB(x...), eachcol(arr))

using Distributions, QuasiMonteCarlo

lb = zeros(3)
ub = ones(3)
n = 10 * 10
u0s = QuasiMonteCarlo.sample(n, lb, ub, LatinHypercubeSample())

function prob_func(prob, i, repeat)
    remake(prob, u0=@view(u0s[:, i]))
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=100, saveat=tspan[1]:0.1:tspan[2])


pixs = map(y -> map(x -> RGB(x...), y), sim)

frame = reshape(first.(pixs), (10, 10))

N = length(pixs[1])

frames = Matrix{RGB}[]
for i in 1:N
    frame = reshape(getindex.(pixs, i), (10, 10))
    frame = map(clamp01nan, frame)
    frame = convert.(RGB{N0f8}, frame)
    push!(frames, frame)
end

for (i, frame) in enumerate(frames)
    save("out2/$i.jpeg", frame)
end

run(`ffmpeg -framerate 60 -start_number 1 -i 'out2/%d.jpeg' -r 60 -y vid/test2.mp4`)
