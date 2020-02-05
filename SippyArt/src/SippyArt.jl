module SippyArt
using WAV, Images
using Flux, Flux.Data.MNIST, Statistics
using Flux: throttle, params
using Distributions
import Distributions: logpdf
using Plots
togray(x) = Gray.(reshape(x, 3024, 3024))
tostereo(x) = reshape(x, (:, 2))

songfn = "/home/sippycups/Music/2019/81 - 9 21 19 2.wav"
wave, sr  = wavread(songfn) # 6,659,596
flatwave = reshape(wave, :)
sr = Int(sr)
println("sample rate: $(sr)")
batch_size = 1
start_sec = 0
end_sec = 23
# sec = wave[sr*start_sec:sr*end_sec - 1, :]
# flatsec = reshape(sec, :)
# out_dim = length(flatsec)
imgfn = "/home/sippycups/audio/square.jpg"
img = load(imgfn)
img_data = channelview(img) # 3, 3024, 3024 == 27,433,728
gray = Gray.(img)
gray_data = channelview(gray) # 3024, 3024 = 9,144,576 first training on gray
smaller = gray_data[988:2011, 988:2011]

gray_floats = convert(Array{Float32}, smaller)
flat_grays = reshape(gray_floats, :)
flat_gray = flat_grays[1:2000]
in_dim = 784 # length(flat_grays)
out_dim = in_dim
out_wav = flatwave[1:in_dim]


bottleneck, mid = 5, 500

# data = zip([flat_grays, out_wav])
data = Iterators.repeated(flat_gray, 1000)

logpdf(b::Bernoulli, y::Bool) = y * log(b.p + eps(Float32)) + (1f0 - y) * log(1 - b.p + eps(Float32))
A, μ, logσ = Dense(in_dim, mid, tanh), Dense(mid, bottleneck), Dense(mid, bottleneck)

# A_out = A(gray_data)
# sig_out = μ(A_out)
# logsig_out = logσ(A_out)
# g_out = g(gray_data)

g(X) = (h = A(X); (μ(h), logσ(h)))

z(μ, logσ) = μ + exp(logσ) * randn(Float32)

f = Chain(Dense(bottleneck, mid, tanh), Dense(mid, out_dim, σ))

kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))

logp_x_z(x, z) = sum(logpdf.(Bernoulli.(f(z)), x))

L̄(X) = ((μ̂, logσ̂) = g(X); (logp_x_z(X, z.(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) * 1 // batch_size)
# L̄(X, Y) = ((μ̂, logσ̂) = g(X); (logp_x_z(Y, z.(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) * 1 // batch_size)

loss(X) = -L̄(X) + 0.01f0 * sum(x->sum(x.^2), params(f))
# loss(X, Y) = -L̄(X, Y) + 0.0001f0 * sum(x->sum(x.^2), params(f))

modelsample() = rand.(Bernoulli.(f(z.(zeros(bottleneck), zeros(bottleneck)))))

# loss(flat_grays, flat_grays)
loss(flat_gray)
loss(out_wav)

evalcb = function () 
	throttle(() -> @show(-L̄(X[:, rand(1:N, batch_size)])), 30)
end

opt = ADAM()
ps = params(A, μ, logσ, f)

Flux.train!(loss, ps, data, opt, cb=evalcb)

# s = randn((88200, 2))
# a = randn((88200, 2))
# s /= maximum(s).^ 2
# vcat(a, s)
# wavwrite(s, "jltest.wav", Fs=44100)


end # module
