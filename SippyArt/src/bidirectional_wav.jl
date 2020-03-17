using WAV
using Flux
using IterTools: ncycle
using Plots
using Glob
dir = "/home/sippycups/Music/2020/"
testdir = "/home/sippycups/Music/2019/"
fns = Glob.glob("*.wav", dir)
testfns = rand(Glob.glob("*.wav", testdir), 5)
fn = fns[1]
n = 1000
hid = 50
epochs = 1 
bs = 64
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

function window_multiple(fns)
    all_xs = Matrix(undef, n, 0)
    all_ys = Matrix(undef, n, 0)
    for f in fns
        xs, ys = window(f)
        all_xs = hcat(all_xs, xs)
        all_ys = hcat(all_ys, ys)
    end
    return all_xs, all_ys
end

xs, ys = window_multiple(fns)
testxs, testys = window_multiple(testfns)

d = Flux.Data.DataLoader(xs, ys; batchsize=bs, shuffle=false)
td = Flux.Data.DataLoader(testxs, testys; batchsize=bs, shuffle=false)
size(d.data[1])
m = Chain(LSTM(n, hid), LSTM(hid, hid), LSTM(hid, n))

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
    i += 1
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
testw = testw[div(length(testw), 2):end]
wavwrite(tw', "train_out8.wav", Fs=sr)
wavwrite(testw', "test_out8.wav", Fs=sr)
 