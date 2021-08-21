using Images, ImageIO, FileIO
using ImageInTerminal, ImageShow
using ColorVectorSpace
using ImageFiltering, ImageContrastAdjustment

function diffusion!(img_new, img, N)
    m, n = size(img)
    for i in 1:m
        for j in 1:n
            if img[i, j] == 0
                continue
            else
                img_new[i, j] = img[i, j]
                for k in 1:N
                    i_new = i + rand(-1:1)
                    j_new = j + rand(-1:1)
                    if i_new < 1 || i_new >= m-1 || j_new < 1 || j_new >= n-1
                        continue
                    else
                        img_new[i_new, j_new] = img[i, j]
                    end
                end
            end
        end
    end
end

function diffusions!(imgs, img, N)
    m, n = size(img)
    for i in 1:m
        for j in 1:n
            if img[i, j] == 0
                continue
            else
                for k in 1:N
                    img_new = imgs[k]
                    img_new[i, j] = img[i, j]
                    i_new = i + rand(-1:1)
                    j_new = j + rand(-1:1)
                    if i_new < 1 || i_new >= m-1 || j_new < 1 || j_new >= n-1
                        continue
                    else
                        img_new[i_new, j_new] = img[i, j]
                    end
                end
            end
        end
    end
end

function diffusion(img, N)
    img_new = similar(img)
    diffusion!(img_new, img, N)
    img_new
end

function animate_diffusion(img, N)
    imgs = [similar(img) for _ in 1:N]
    diffusions!(imgs, img, N)
    imgs
end

function animate_filt(img, N)
    imgs = [similar(img) for _ in 1:N]
    for i in 1:N
        # diffusions!(imgs[i], img, N)
        imfilter!(imgs[i], img, Kernel.gaussian(i));
        # imfilter!(imgs[i], img, Kernel.sobel());
    end
    imgs
end

function anim(imgs; sleep_amt=0.1)
    for i in 1:length(imgs)
        display(imgs[i])
        sleep(sleep_amt)
    end
end

img = load("/Users/anand/Pictures/81.jpeg")
img = load("/Users/anand/Pictures/81.jpeg")
img = load("pic.jpg")
img = ColorVectorSpace.complement.(img)


alg = Equalization(nbins = 256)
img = adjust_histogram(img, alg)

imgs = animate_diffusion(img, 1000)
imgs = animate_filt(img, 100)
anim(imgs)

for (i, img) in enumerate(imgs)
    save("outs/$i.jpeg", img)
end

run(`ffmpeg -framerate 60 -start_number 1 -i 'out2/%d.jpeg' -r 60 -y vid/test2.mp4`)


imgs[1] imgs[end] == imgs[1]

imgg = imfilter(img_adjusted, Kernel.gaussian(3));
imgg = imfilter(imgg, Kernel.gaussian(100))

