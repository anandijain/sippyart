using WAV, Flux
using IterTools: ncycle
fn = "/home/sippycups/Music/2020/81 - 3 14 20 2.wav"
testfn = "/home/sippycups/Music/2020/81 - 3 14 20.wav"
n = 2000
epochs = 10
bs = 128

function window(fn::String)
    w = wavread(fn)[1]
    c = w[:, 1]
    len = size(w)[1] - (2n + 1)
    m = div(len, n)
    idxs = 1:2n:m*n
    xs = hcat([vec(c[i:i+n-1]) for i in idxs]...)
    ys = hcat([vec(c[i+n:i+2n-1]) for i in idxs]...)

    return Flux.Data.DataLoader(xs, ys; batchsize=bs, shuffle=false)
end

d = window(fn)
td = window(testfn)

m = Chain(LSTM(n, 100), LSTM(100, 100), LSTM(100, n))

function lxss(x, y)
    l = sum(Flux.mse.(m(x), y))
    Flux.reset!(m)
    return l
end

ps = params(m)
opt = ADAM()
m(d.data[1][2])
# lxss(d.data[1], d.data[2])
# lxss(td.data[1][2], td.data[2][2])
function getfirst()
    for (x, y) in d
        return x, y
    end
end
x, y = getfirst()
lxss(x, y)

train_pred = Vector()
test_pred = Vector()
train_losses = Vector()
test_losses = Vector()
cb = function ()  
    for (x, y) in td
        println("x: ", size(x))
        println("y: ", size(y))
        Flux.reset!(m)
        @assert size(x) == (2000, bs)
        @assert size(y) == (2000, bs)
        test_ŷ = m(x)

        Flux.reset!(m)
        test_loss = lxss(x, y)
        println(test_loss)
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
        println(train_loss)
        Flux.reset!(m)

        push!(train_pred, ŷ)
        push!(train_losses, train_loss)
        break
    end
end
for e in 1:epochs
    println("new epoch: ", e)
    Flux.train!(lxss, ps, d, opt, cb=cb)
end

to_write = vcat(train_pred...)
wavwrite(to_write, "test_out.wav", Fs=44100)
