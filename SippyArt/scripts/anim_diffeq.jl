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

using WAV
function f(du,u,p,t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2] #prey
  du[2] = -p[3]*u[2] + p[4]*u[1]*u[2] #predator
end

u0 = [1.0;1.0]
p = [1.5,1.0,3.0,1.0]
tspan = (0.0, 1000.0)
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob; saveat=tspan[1]:0.001:tspan[2])
arr = Array(sol)
for i in 1:length(u0)
    arr[i, :] .= arr[i, :] ./ maximum(arr[i, :])
end
arr = 2*arr .- 1
wavwrite(arr', 44100, "lotka.wav")
w, sr = wavread("/Users/anand/ableton_files/081821081821.wav")
w, sr = wavread("/Users/anand/ableton_files/081821 Project/081821.wav")

function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

p = [10.0,28.0,8 / 3]
u0 = [1.0;0.0;0.0]
tspan = (0.0, 1000.0)
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob; saveat=tspan[1]:0.001:tspan[2])
arr = Array(sol)
for i in 1:length(u0)
    arr[i, :] .= arr[i, :] ./ maximum(arr[i, :])
end
arr = 2*arr .- 1
wavwrite(arr[1:2, :]', 44100, "foo.wav")

using Catalyst, DifferentialEquations, WAV

repressilator = @reaction_network begin
    hillr(P₃,α,K,n), ∅ --> m₁
    hillr(P₁,α,K,n), ∅ --> m₂
    hillr(P₂,α,K,n), ∅ --> m₃
    (δ,γ), m₁ ↔ ∅
    (δ,γ), m₂ ↔ ∅
    (δ,γ), m₃ ↔ ∅
    β, m₁ --> m₁ + P₁
    β, m₂ --> m₂ + P₂
    β, m₃ --> m₃ + P₃
    μ, P₁ --> ∅
    μ, P₂ --> ∅
    μ, P₃ --> ∅
end α K n δ γ β μ;

odesys = convert(ODESystem, repressilator)

# parameters [α,K,n,δ,γ,β,μ]
p = (.5, 40, 2, log(2)/120, 5e-3, 20*log(2)/120, log(2)/60)

# initial condition [m₁,m₂,m₃,P₁,P₂,P₃]
u0 = [0.,0.,0.,20.,0.,0.]

# time interval to solve on
tspan = (0., 100000.)

# create the ODEProblem we want to solve
prob = ODEProblem(repressilator, u0, tspan, p)
sol = solve(prob; saveat=tspan[1]:0.1:tspan[2])
arr = Array(sol)
for i in 1:length(u0)
    arr[i, :] .= arr[i, :] ./ maximum(arr[i, :])
end
arr = 2*arr .- 1
for i in 1:2:length(u0)-1
    wavwrite(arr[i:i+1, :]', 44100, "repressilator3_$i.wav")
end

# redefine the initial condition to be integer valued
u₀ = [0,0,0,20,0,0]

# next we create a discrete problem to encode that our species are integer valued:
dprob = DiscreteProblem(repressilator, u₀, tspan, p)

# now, we create a JumpProblem, and specify Gillespie's Direct Method as the solver:
jprob = JumpProblem(repressilator, dprob, Direct(), save_positions=(false,false))

sol = solve(jprob, SSAStepper(); saveat=tspan[1]:0.1:tspan[2])
arr = Float64.(Array(sol))
for i in 1:length(u0)
    arr[i, :] .= arr[i, :] ./ maximum(arr[i, :])
end
arr = 2*arr .- 1
for i in 1:2:length(u0)-1
    wavwrite(arr[i:i+1, :]', 44100, "data/repressilator_jump_3_$i.wav")
end