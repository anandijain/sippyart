using WAV, Flux
using IterTools: ncycle
fn = "/home/sippycups/Music/2020/81 - 3 14 20 2.wav"

function window(fn::String;n=100)
    w = wavread(fn)[1]
    c = w[:, 1]
    len = size(w)[1] - (2n + 1)
    m = div(len, n)
    idxs = 1:2n:m*n
    xs = Vector([vec(c[i:i+n-1]) for i in idxs])
    ys = Vector([vec(c[i+n:i+2n-1]) for i in idxs])

    return Flux.Data.DataLoader(xs, ys; batchsize=1)
end

d = window(fn)
m = Chain(LSTM(100, 100), LSTM(100, 100), LSTM(100, 100))

function lxss(x, y) 
    l = sum(Flux.mse.(m(x[1]), y[1]))
    Flux.reset!(m)
    return l
end

ps = params(m)
opt = ADAM()
m(d.data[1][2])
lxss(d.data[1][2], d.data[2][2])
function getfirst()
    for (x, y) in d
        return x, y
        break
        println(y)
        println(size(y[1]))
        println(lxss(x, y))
    end
end
x, y = getfirst()
lxss(x, y)

Flux.train!(lxss, ps, ncycle(d, 10), opt)

