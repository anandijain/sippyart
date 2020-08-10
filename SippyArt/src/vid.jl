using WAV, Images, VideoIO, Colors
using Base.Iterators, Random, StatsBase

# attempt at non trash code lol
# simple objective: random isomorphism between wav file and an mp4 video
# given a wav file creates a music video that preserves the structure of the bits 

function frames(fname; fps=60) #::Union{Nothing, Any}
	s, fs = wavread(fname)
	frames(s, fs, fps) 
end 

frames(s, fs, fps) = partition(eachrow(s), Int(fs / fps))

"dict from samples to grey shades of type RGB (idkwhylol)"
function gmap(s) 
	us = unique(s)
	Dict(us .=> range(RGB(0), RGB(1), length=length(us)))
end	

"dict colormap, but written dumb, how do i sample from multiple ranges"
function cmap(s; )
	us = unique(s)
	n = length(us)
	Dict(us .=> sample(all_colors_fast(), n, replace=false))
end


"horrible function for getting all colors, doesn't use rand. dont use this" 
function all_colors() 
	arr = RGB[]
	for i in 1:255
		for j in 1:255
			for k in 1:255
				f = [i, j, k] ./ 255
				push!(arr, RGB(f...))
			end
		end
	end
	arr
end

# thanks kimikage
mutable struct Xorshift24 <: AbstractRNG
    prev::UInt32
    Xorshift24(seed::Integer) = new(seed % 0xFFFFFF + 1) 
end

function Random.rand(rng::Xorshift24, ::Random.SamplerType{UInt32}) # The parameters are just an example.  I don't know the randomness.
    x = rng.prev
    x ⊻= x >> 13
    x ⊻= (x << 19) >> 8
    x ⊻= x >> 2
    rng.prev = x
end

"better version of all_colors"
all_colors_fast() = reinterpret(RGB24, rand(Xorshift24(0), UInt32, 2^24 - 1))

function imgs_from_signal(s; fs=44_100, fps=60)
	d = cmap(s)
	map(x->reshape(hcat(x...), 30, 49), collect(frames(map(x->d[x], s), fs, fps))[1:end-1])
end

# todo: use AxisArrays for the times
# write sound and images into mp4
# save imgs, seems like FileIO was being a little jank. 

# might use Plots mp4() to write then just add the sound on top? idk

# TEST_PATH="data/072520.wav"


function main(wavpath)
	rootname = split(split(wavpath, "/")[end], ".")[1]	
	outname = "data/imgs"
	s, fs = wavread(wavpath)
	imgs = imgs_from_signal(s)
	if isdir(outname)
		error("folder exists brooo")
	else 
		mkpath(outname)
		map(x->save("$outname/$(x[1]).png", x[2]), enumerate(imgs))
	end
end

main(ARGS[1])