module Utils
using WAV

function window(n::Int, sr::Int, fn::String)
    w = wavread(fn)[1]
    c = w[1:20*sr, 1]
    len = size(c)[1] - (2n + 1)
    m = div(len, n)
    idxs = 1:2n:m*n
    xs = hcat([vec(c[i:i+n-1]) for i in idxs]...)
    ys = hcat([vec(c[i+n:i+2n-1]) for i in idxs]...)
    return xs, ys
end

function window_multiple(n::Int, sr::Int, fns::Vector{String})
    all_xs = Matrix(undef, n, 0)
    all_ys = Matrix(undef, n, 0)
    for f in fns
        xs, ys = window(n, sr, f)
        all_xs = hcat(all_xs, xs)
        all_ys = hcat(all_ys, ys)
    end
    return all_xs, all_ys
end
end