using WAV, Flux
using IterTools: ncycle
using Plots

fn = "/home/sippycups/Music/2020/81 - 3 14 20 2.wav"
testfn = "/home/sippycups/Music/2020/81 - 3 14 20.wav"
n = 2205
epochs = 3 
bs = 128
sr = 44100

function window(fn::String)
    w = wavread(fn)[1]
    c = w[1:20*sr, 1]
    len = size(c)[1] - (2n + 1)
    m = div(len, n)
    idxs = 1:2n:m*n
    xs = hcat([vec(c[i:i+n-1]) for i in idxs]...)
    ys = hcat([vec(c[i+n:i+2n-1]) for i in idxs]...)
    return xs, ys
end

d = Flux.Data.DataLoader(window(fn)...; batchsize=bs, shuffle=false)
td = Flux.Data.DataLoader(window(testfn)...; batchsize=bs, shuffle=false)

m = Chain(LSTM(n, 500),
#  LSTM(200, 200),
LSTM(500, n))

function lxss(x, y)
    @assert size(x) == (n, bs)
    @assert size(y) == (n, bs)
    l = sum(Flux.mse.(m(x), y))
    Flux.reset!(m)
    return l
end

ps = params(m)
opt = ADAM()


train_pred = Vector()
test_pred = Vector()
train_losses = Vector()
test_losses = Vector()
cb = function ()  
    for (x, y) in td
        Flux.reset!(m)
        @assert size(x) == (n, bs)
        @assert size(y) == (n, bs)
        test_ŷ = m(x)

        Flux.reset!(m)
        test_loss = lxss(x, y)
        println("test_loss:", test_loss)
        Flux.reset!(m)
        
        push!(test_pred, test_ŷ)
        push!(test_losses, test_loss)
        break
    end
    for (x, y) in d
        Flux.reset!(m)
        ŷ = m(x[1])
        Flux.reset!(m)
        train_loss = lxss(x, y)
        println("train_loss:", train_loss)
        Flux.reset!(m)

        push!(train_pred, ŷ)
        push!(train_losses, train_loss)
        break
    end
end
for e in 1:epochs
    println("new epoch: ", e)
    Flux.train!(lxss, ps, td, opt, cb=cb)
end

to_write = vcat(train_pred...)
tw = reshape(to_write, 1, :)
size(tw)[2] / sr # seconds

tw = tw[9*div(length(tw), 10):end]
test_write = vcat(test_pred...)
testw = reshape(test_write, 1, :)
testw = testw[div(length(testw), 2):end]
wavwrite(tw, "train_out6.wav", Fs=sr)
wavwrite(testw, "test_out6.wav", Fs=sr)
 