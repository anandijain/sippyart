using WAV, Images
using Base.Iterators, Random


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
function cmap(s)
	us = unique(s)
	n = length(us)
	Dict(us .=> sample(shuffle(all_colors()), n, replace=false))
end


"horrible function for getting all colors, doesn't use rand" 
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


# todo: use AxisArrays for the times
# write sound and images into mp4
