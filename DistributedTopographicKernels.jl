using Flux
using SpecialFunctions
using Distributions
using CUDA
using LinearAlgebra

module DistributedTopographicKernels
export TopographicKernel
export rainbow_plot_kernel
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Constructor Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
struct TopographicKernel
    kernel::Array{Float64, 4}
    function TopographicKernel(N, nkern, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; eta=0.1, hyp1=0.9, hyp2=0.99, seed_number=1)
        # initialise on the GPU
        xret, yret = retinal_initialisation(N, nkern)
        xs, ys = collicular_initialisation(xret, yret)
        gpu_xret, gpu_yret, gpu_xs, gpu_ys = CuArray.([xret, yret, xs, ys])

        # minimise using .Flux and transfer back to cpu
        gpu_xs_final, gpu_ys_final = minimise(gpu_xs, gpu_ys, T; s, alpha, beta, gamma, delta, sigmacol, sigmaret, gpu_xret, gpu_yret, eta, hyp1, hyp2)
        xs_final = Array(xs)
        ys_final = Array(ys)
        
        # sample 
        kernel = synaptic_sampling(xs_final, ys_final, s, xret, yret, nkern, ncontacts; seed_number) 
        new(kernel)
    end
end

function retinal_initialisation(N, nkern)
    L = round(Int, sqrt(N / nkern))
    xret = zeros(N)
    yret = zeros(N)
    for i = 1:nkern:N
        ind = floor(Int, i/nkern)
        xret[i:(i + nkern - 1)] = mod(ind, L)
        yret[i:(i + nkern - 1)] = floor(ind / L)
    end
    return xret, yret
end

function collicular_initialisation(xret, yret)
    return 0.25 .* xret, 0.25 .* yret
end

function synaptic_sampling(xs, ys, s, xret, yret, nkern, ncontacts; seed_number=1) 
    Random.seed!(seed_number) 
    for i = 1:nkern:length(xs)
        sigma = s * Matrix(I, 2, 2)
        pdf_i = Distributions.MixtureModel(
            MvNormal[
                [MvNormal([xs[p], ys[p]], sigma) for p in i:(i + nkern- 1)]... 
            ]
        )
        for j = 1:ncontacts
            xj, yj = sample(pdf_i)
            ind = floor(i/nkern) * ncontacts + j

            array[ind, 1] = xret[i]
            array[ind, 2] = yret[i]
            array[ind, 3] = xj
            array[ind, 4] = yj
        end
    end
    return array
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Energy Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function echemi(xi, yi, s, alpha, beta)
    a = alpha * (erf(1/s * (1 - yi)) + erf(yi / s)) * (erf(1/(2*s) * (2 + s^2 - 2 * xi)) + erf(1/(2*s) * (-2 + s^2 - 2 * xi)) - erf(s/2 - xi/s) - erf(s/2 + xi/s))
    b = beta * (erf(1/s * (1 - xi)) + erf(xi / s)) * (erf(1/(2*s) * (2 + s^2 - 2 * yi)) + erf(1/(2*s) * (-2 + s^2 - 2 * yi)) - erf(s/2 - yi/s) - erf(s/2 + yi/s)))
    return 0.25 * exp(s^2/4) * (b - a)
end

function eactij(xi, yi, xj, yj, s, xreti, yreti, xretj, yretj, sigmacol, sigmaret, gamma)
    a = (xj^2 + yj^2)/s^2 + ((xi-xj)^2+(yi-yj)^2)/sigmacol^2 + ((xreti-xretj)^2 + (yreti-yretj)^2)/sigmaret^2
    b = (erf((2-xj)/(sqrt(2)*s)) + erf(xj/(sqrt(2)*s))) * (erf((2-yj)/(sqrt(2)*s)) + erf(yj/(sqrt(2)*s)))
    return gamma/8 * exp(-1/2 * a) * b
end

function ecompij(xi, yi, xj, yj, s, delta)
    return 1/8 * exp(- 1/2 * ((xi-xj)^2+(yi-yj)^2)/sigmacol^2) * (erf((-2+xi+xj)/(sqrt(2)*s)) - erf((xi+xj)/(sqrt(2)*s))) * (erf((-2+yi+yj)/(sqrt(2)*s)) - erf((yi+yj)/(sqrt(2)*s)))
end

function ei(i, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xs, ys, xret, yret)
    chem = echemi(col_xy[i, 1], col_xy[i, 2], s, alpha, beta)
    act = sum((xj, yj, xretj, yretj) -> eactij(xs[i], ys[i], xj, yj, s, xret[i], xret[i], xretj, yretj, sigmacol, sigmaret, gamma), xs, ys, xret, yret)
    comp = sum((xj, yj) -> eactij(xs[i], ys[i], xj, yj, s), xs, ys)
    return chem + act + comp
end

function energy(xs, ys; s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret)
    return sum(i -> ei(i, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xs, ys, xret, yret), 1:length(xs))
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Minimisation.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function minimise(xs, ys, T; s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, eta=0.01, hyp1=0.9, hyp2=0.99)
    opt = Flux.Optimise.ADAM(eta, (hyp1, hyp2))
    loss = energy(xs, ys; s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret)
    for t = 1:T
        θ = Params([xs, ys])
        θ̄bar= gradient(() -> loss(), θ)
        Flux.Optimise.update!(opt, θ, θ̄bar)
    end
    return xs, ys
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

function rainbow_plot_kernel(kernel::TopographicKernel; pal=0.45, sz=0.1, dpi = 400)
    x_ret = kernel[:, 1]
    y_ret = kernel[:, 2]
    x_col = kernel[:, 3]
    y_col = kernel[:, 4]

    cols = map((x, y) -> RGBA(x, pal, y, 1), x_ret, y_ret)

    plt1 = scatter(x_ret, y_ret, color=cols, markersize=sz, xlabel="Scaled Nasal Field", ylabel="Scaled Temporal Field")
    plt2 = scatter(x_col, y_col, color=cols, markersize=sz, ylabel="Scaled Rostral Field", ylabel="Scaled Caudal Field")
    final = plot(plt1, plt2, layout=(2,1), title="Topographic Projection", dpi=DPI)
    return final
end

