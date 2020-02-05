using WAV
using DifferentialEquations, Flux, DiffEqFlux, Plots
using StochasticDiffEq
using DiffEqBase.EnsembleAnalysis
using StatsBase, Statistics

songfn = "/home/sippycups/Music/2019/81 - 9 21 19 2.wav"
wave, sr = wavread(songfn)
start = 5
secs = 5
start_idx = Int(sr*start)
cut_idx = start_idx + Int(sr*secs)
t = float(start_idx):float(cut_idx)
# tspan = (0.0, 1.0)
target_data = wave[start_idx:cut_idx, 2:end]'
u0 = target_data[:, 1]
dim = length(u0)

sde_data_vars = zeros(dim, length(t)) .+ 1e-3

drift_dudt = Chain(
    Dense(dim, 20, tanh),
    # Dense(20, 20, tanh),
    Dense(20, 20, tanh),
    Dense(20, dim)
) #|> gpu

diffusion_dudt = Chain(Dense(dim,dim))

n_sde = NeuralDSDE(drift_dudt,diffusion_dudt,(t[1], t[end]),SOSRI(),saveat=t,reltol=1e-1,abstol=1e-1)
ps = Flux.params(n_sde)
pred = n_sde(u0)

function predict_n_sde()
	Array(n_sde(u0))
end

function loss_n_sde(;n=5)
	samples = [predict_n_sde() for i in 1:n]
	means = reshape(mean.([[samples[i][j] for i in 1:length(samples)] for j in 1:length(samples[1])]),size(samples[1])...)
	vars = reshape(var.([[samples[i][j] for i in 1:length(samples)] for j in 1:length(samples[1])]),size(samples[1])...)
	sum(abs2,target_data - means) + sum(abs2,sde_data_vars - vars)  # try to replace sde_data_vars with flux_err
end

repeated_data = Iterators.repeated((), 1000)
opt = ADAM(0.025)

losses = []
cb = function ()
	cur_pred = predict_n_sde()
	display(string("pred : ", cur_pred[1:2, 2:end]))
	display(string("target : ", target_data[1:2, 2:end]))
	pl = scatter(t[2:end], target_data[1, 2:end], label="flux_target", markersize=5, markercolor=:blue)
	scatter!(pl, t[2:end], target_data[2, 2:end], label="flux_err_target", markersize=5, markercolor=:green)
	scatter!(pl, t[2:end], cur_pred[1, 2:end], label="flux_pred", markersize=5, markercolor=:red) #, markercolor=:red)
	scatter!(pl, t[2:end], cur_pred[2, 2:end], label="flux_err_pred", markersize=5, markercolor=:orange)
	yticks!([-5:5;])
	xticks!(t[2]:t[end])
	plot!(pl, xlabel="normed_mjd", ylabel="normed_param", size=(900, 900))
end
cb()
Flux.train!(loss_n_sde, ps, repeated_data, opt, cb = cb)

end
