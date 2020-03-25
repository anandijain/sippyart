using WAV, Plots, Glob, Distributions
using Flux
using IterTools: ncycle
# import Distributions: logpdf
using Flux: throttle, params

include("utils.jl")
using .Utils

dir = "/home/sippycups/Music/2020/"
testdir = "/home/sippycups/Music/2019/"
fns = Glob.glob("*.wav", dir)
testfns = rand(Glob.glob("*.wav", testdir), 1)
fn = fns[1]
n, bs = 100, 64
Dz, Dh = 50, 100

epochs = 1 
sr = 44100

xs, ys = Utils.window_multiple(n, sr, fns)
testxs, testys = Utils.window_multiple(n, sr, testfns)

d = Flux.Data.DataLoader(xs, ys; batchsize=bs, shuffle=false)
td = Flux.Data.DataLoader(testxs, testys; batchsize=bs, shuffle=false)
size(d.data[1])

m = Chain(LSTM(n, Dz), LSTM(Dz, Dz), LSTM(Dz, Dz), LSTM(Dz, n)) # basic

A, μ, logσ = Chain(x-> σ.(x),  LSTM(n, Dh)), Dense(Dh, Dz), Dense(Dh, Dz)
g(X) = (h = A(X); (μ(h), logσ(h)))
z(μ, logσ) = μ + exp(logσ) * randn(Float32)
f = Chain(Dense(Dz, Dh, tanh), LSTM(Dh, n), x ->σ.(x))
kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))
logpdf(b::Bernoulli, y::Bool) = y * log(b.p + eps(Float32)) + (1f0 - y) * log(1 - b.p + eps(Float32))
logp_x_z(x, z) = sum(logpdf.(Bernoulli.(f(z)), x))
L̄(X) = ((μ̂, logσ̂) = g(X); (logp_x_z(X, z.(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) * 1 // bs)
loss(X) = -L̄(X) + 0.01f0 * sum(x->sum(x.^2), params(f))
modelsample() = rand.(Bernoulli.(f(z.(zeros(Dz), zeros(Dz)))))

evalcb = throttle(() -> @show(-L̄(d.data[1][:, rand(1:n, bs)])), 30)
ps = params(A, μ, logσ, f)

function lxss(x, y)
    @assert size(x) == (n, bs)
    @assert size(y) == (n, bs)
    l = sum(Flux.mse.(m(x), y))
    Flux.reset!(m)
    return l
end

ps = params(m)
opt = ADAM()

function eval_loader(l::Flux.Data.DataLoader)
    for (x, y) in l
        @assert size(x) == (n, bs)
        @assert size(y) == (n, bs)
        ŷ = m(x)

        loss = lxss(x, y)
        return ŷ, loss
    end
end

train_pred = Vector()
test_pred = Vector()
train_losses = Vector()
test_losses = Vector()

global i = 1
cb = function ()  
    ŷ, loss = eval_loader(d)
    push!(train_pred, ŷ)
    push!(train_losses, loss)

    test_ŷ, test_loss = eval_loader(td)
    push!(test_pred, test_ŷ)
    push!(test_losses, test_loss)
    println("iter: ", i, ": train_loss:", loss, " test_loss:", test_loss)
    global i += 1
end

for e in 1:epochs
    println("new epoch: ", e)
    Flux.train!(lxss, ps, d, opt, cb=cb)
end

to_write = vcat(train_pred...)
tw = reshape(to_write, 1, :)
size(tw)[2] / sr # seconds

# tw = tw[div(length(tw), 1):end]
test_write = vcat(test_pred...)
testw = reshape(test_write, 1, :)
# testw = testw[div(length(testw), 2):end]
wavwrite(tw', "train_out10.wav", Fs=sr)
wavwrite(testw', "test_out10.wav", Fs=sr)
 