
module DistributedTopographicKernels
using Flux
using SpecialFunctions
using Distributions
using CUDA
using LinearAlgebra
using Random
using Plots
using ProgressMeter


export TopographicKernel
export rainbow_plot_kernel
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Constructor Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
struct TopographicKernel
    kernel::Array{Float64, 2}
    rtime::Float64
    ttime::Float64
    function TopographicKernel(N, nkern, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; eta=0.01, hyp1=0.9, hyp2=0.999, case="WT", seed_number=1, epha3_level=1.0, epha3_fraction=0.5)
        # initialise on the GPU
        CUDA.device!(1)
        xret, yret = retinal_initialisation(N, nkern)
        xs, ys = collicular_initialisation(xret, yret)
        epha3_mask = map(x -> rand() < epha3_fraction, 1:length(xret))
        gpu_xret, gpu_yret, gpu_xs, gpu_ys, gpu_epha3_mask = CUDA.CuArray.([xret, yret, xs, ys, epha3_mask])

        # minimise using .Flux and transfer back to cpu
        t1 = @elapsed gpu_xs_final, gpu_ys_final = gpu_optimise(gpu_xs, gpu_ys, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, gpu_xret, gpu_yret, eta, hyp1, hyp2, gpu_epha3_mask, epha3_level, case)
        t2 = @elapsed theoretical_timer(gpu_xs, gpu_ys, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, gpu_xret, gpu_yret, eta, hyp1, hyp2, gpu_epha3_mask, epha3_level, case)
        xs_final = Array(gpu_xs_final)
        ys_final = Array(gpu_ys_final)
        CUDA.unsafe_free!.([gpu_xs_final, gpu_ys_final, gpu_xret, gpu_yret, gpu_xs, gpu_ys])
        CUDA.reclaim()

        # sample 
        kernel = synaptic_sampling(xs_final, ys_final, s, xret, yret, nkern, ncontacts; seed_number, epha3=epha3_mask) 

        # a theoretical timing bound is 4 times the energy call
        
        new(kernel, t1, 4 * t2)
    end
end

function retinal_initialisation(N, nkern)
    L = round(Int, sqrt(N / nkern))
    xret = zeros(Float32, N)
    yret = zeros(Float32, N)
    for i = 1:nkern:N
        ind = floor(Int, i/nkern)
        imax = maximum([i + nkern - 1, N])
        xret[i:imax] .= mod(ind, L) / L .+ 1f-7
        yret[i:imax] .= floor(ind / L) / L .+ 1f-7
    end
    return xret, yret
end

function collicular_initialisation(xret, yret)
    x = 0.0 .+ 0.35 .* xret .+ 0.65 .* (rand(length(xret)))
    y = 0.0 .+ 0.35 .* yret .+ 0.65 .* (rand(length(xret)))
    return x, y
end

function synaptic_sampling(xs, ys, s, xret, yret, nkern, ncontacts; seed_number=1, epha3=zeros(length(xs))) 
    Random.seed!(seed_number)
    array = zeros(ceil(Int, length(xs)/nkern) * ncontacts, 5)
    for i = 1:nkern:length(xs)
        sigma = 0.001 .* Matrix(I, 2, 2)
        pdf_i = Distributions.MixtureModel(
            MvNormal[
                [MvNormal([xs[p], ys[p]], sigma) for p in i:(i + nkern- 1)]... 
            ]
        )
        if epha3[i] == 1
            tag_epha3 = 1
        else
            tag_epha3 = 0
        end

        for j = 1:ncontacts
            xj, yj = rand(pdf_i) #  [xs[i], ys[i]]# 
            ind = floor(Int, (i-1)/nkern) * ncontacts + j

            array[ind, 1] = xret[i]
            array[ind, 2] = yret[i]
            array[ind, 3] = xj
            array[ind, 4] = yj
            array[ind, 5] = tag_epha3
        end
    end
    return array
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Energy Functions.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function echemi(xi, yi, s, alpha, beta, xret, yret)
    K = 1.0
    a = alpha .* (1.75 .* exp.(K .* (xi .- 0.5)) .* exp.(K .* (-xret .+ 0.5)) .+ exp.(K .* (xret .- 0.5)) .* exp.(K .* (-xi .+ 0.5))) # alpha .* (erf.((1 .- yi) ./ s) .+ erf.(yi ./ s)) .* (exp.(xret .- xi) .* (erf.((2 .+ s^2 .- 2 .* xi) ./ (2*s)) .+ erf.(s/2 .- xi ./ s)) .+ exp.(xi .- xret) .* (erf.((-2 .+ s^2 .- 2 .* xi) ./ (2*s)) .+ erf.(s/2 .+ xi ./ s))) # 
    b = beta .* (1.75 .* exp.(K .* (yi .- 0.5)) .* exp.(K .* (-yret .+ 0.5)) .+ exp.(K .* (yret .- 0.5)) .* exp.(K .* (-yi .+ 0.5))) # beta .* (erf.((1 .- xi) ./ s) .+ erf.(xi ./ s)) .* (exp.(yret .- yi) .* (erf.((2 .+ s^2 .- 2 .* yi) ./ (2*s)) .+ erf.(s/2 .- yi ./ s)) .+ exp.(yi .- yret) .* (erf.((-2 .+ s^2 .- 2 .* yi) ./ (2*s)) .+ erf.(s/2 .+ yi ./ s))) # 
    return  a .+ b # 0.25 * exp(s^2/4) .* (b .+ a)  #
end

function echem_ephrinA2A5(xi, yi, s, alpha, beta, xret, yret)
    K = 1.0
    a = alpha .* (1.75 .* exp.(K .* (xi .- 0.5)) .* exp.(K .* (-xret .+ 0.5)) .+ exp.(K .* (xret .- 0.5)) .* exp.(K .* (-xi .+ 0.5))) #a = alpha .* (1.5 .* exp.(xi .- 0.5) .* exp.(-xret .+ 0.5) .+ exp.(xret .- 0.5) .* exp.(-xi .+ 0.5)) # a = alpha .* (erf.((1 .- yi) ./ s) .+ erf.(yi ./ s)) .* (exp.(xret .- xi) .* (erf.((2 .+ s^2 .- 2 .* xi) ./ (2*s)) .+ erf.(s/2 .- xi ./ s)) .+ exp.(xi .- xret) .* (erf.((-2 .+ s^2 .- 2 .* xi) ./ (2*s)) .+ erf.(s/2 .+ xi ./ s))) # 
    b = beta .* (1.75 .* exp.(K .* (yi .- 0.5)) .* exp.(K .* (-yret .+ 0.5)) .+ exp.(K .* (yret .- 0.5)) .* exp.(K .* (-yi .+ 0.5))) # b = beta .* (1.5 .* exp.(yi .- 0.5) .* exp.(-yret .+ 0.5) .+ exp.(yret .- 0.5) .* exp.(-yi .+ 0.5)) # b =  beta .* (erf.((1 .- xi) ./ s) .+ erf.(xi ./ s)) .* (exp.(yret .- yi) .* (erf.((2 .+ s^2 .- 2 .* yi) ./ (2*s)) .- erf.(s/2 .- yi ./ s)) .+ exp.(yi .- yret) .* (erf.((-2 .+ s^2 .- 2 .* yi) ./ (2*s)) .- erf.(s/2 .+ yi ./ s))) # 
    return b # 0.25 * exp(s^2/4) .* (b)# .+ a .* (xret .< 0.05)) 
end

function echem_ephA3(xi, yi, s, alpha, beta, xret, yret, epha3_mask, epha3_level)
    K = 1.0
    a = alpha .* (1.75 .* exp.(K .* (xi .- 0.5)) .* exp.(K .* (-xret .+ 0.5)) .+ exp.(K .* (xret .- 0.5)) .* exp.(K .* (-xi .+ 0.5)))# a = alpha .* (1.5 .* exp.(xi .- 0.5) .* exp.(-xret .+ 0.5) .+ exp.(xret .- 0.5) .* exp.(-xi .+ 0.5)) # alpha .* (erf.((1 .- yi) ./ s) .+ erf.(yi ./ s)) .* (exp.(xret .- xi) .* (erf.((2 .+ s^2 .- 2 .* xi) ./ (2*s)) .- erf.(s/2 .- xi ./ s)) .+ exp.(xi .- xret) .* (erf.((-2 .+ s^2 .- 2 .* xi) ./ (2*s)) .- erf.(s/2 .+ xi ./ s))) # 
    epha3 = epha3_mask .* epha3_level .* exp.(K .* (xi .- 0.5)) .* exp.(K .* (-xret .+ 0.5)) # epha3 = epha3_mask .* epha3_level .* exp.(xret .- 0.5) .* exp.(-xi .+ 0.5)# epha3_mask .* epha3_level .* (erf.((1 .- yi) ./ s) .+ erf.(yi ./ s)) .* (exp.(xret .- xi) .* (erf.((2 .+ s^2 .- 2 .* xi) ./ (2*s)) .- erf.(s/2 .- xi ./ s))) # 
    b = beta .* (1.75 .* exp.(K .* (yi .- 0.5)) .* exp.(K .* (-yret .+ 0.5)) .+ exp.(K .* (yret .- 0.5)) .* exp.(K .* (-yi .+ 0.5))) # b = beta .* (1.5 .* exp.(yi .- 0.5) .* exp.(-yret .+ 0.5) .+ exp.(yret .- 0.5) .* exp.(-yi .+ 0.5)) # beta .* (erf.((1 .- xi) ./ s) .+ erf.(xi ./ s)) .* (exp.(yret .- yi) .* (erf.((2 .+ s^2 .- 2 .* yi) ./ (2*s)) .- erf.(s/2 .- yi ./ s)) .+ exp.(yi .- yret) .* (erf.((-2 .+ s^2 .- 2 .* yi) ./ (2*s)) .- erf.(s/2 .+ yi ./ s))) # 
    return b .+ a .+ epha3 # 0.25 * exp(s^2/4) .* (b .+ a .+ epha3) 
end

function eactij(xi, yi, xj, yj, s, xreti, yreti, xretj, yretj, sigmacol, sigmaret, gamma)
    a = (xj.^2 .+ yj'.^2) ./ s^2 .+ ((xi .- xj').^2 .+ (yi .- yj').^2) ./ sigmacol^2 .+ ((xreti .- xretj').^2 + (yreti .- yretj').^2) ./ sigmaret^2
    b = (erf.((2 .- xj) ./ (sqrt(2)*s)) .+ erf.(xj' ./ (sqrt(2)*s))) .* (erf.((2 .- yj) ./ (sqrt(2)*s)) .+ erf.(yj' ./ (sqrt(2)*s)))
    
    return gamma/8 * exp.(-1/2 .* a) .* b
end

function ecompij(xi, yi, xj, yj, s, delta)
    return delta/8 .* exp.(- 1/2 .* ((xi.-xj').^2 .+ (yi.-yj').^2) ./ s^2) .* (erf.((-2 .+ xi .+ xj') ./ (sqrt(2)*s)) .- erf.((xi .+ xj') ./ (sqrt(2)*s))) .* (erf.((-2 .+ yi .+ yj') ./ (sqrt(2)*s)) .- erf.((yi .+ yj') ./ (sqrt(2)*s)))
end

function energy(xs, ys; s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, epha3_mask, epha3_level, case)
    if case == "WT"
        chem = (echemi(xs, ys, s, alpha, beta, xret, yret)) ./ length(xs)
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, sigmaret, gamma), dims=2) ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, s, delta), dims=2) ./ length(xs)^2
    elseif case == "ephrinA2A5"
        chem = echem_ephrinA2A5(xs, ys, s, alpha, beta, xret, yret) ./ length(xs)
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, sigmaret, gamma), dims=2) ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, s, delta), dims=2) ./ length(xs)^2
    elseif case == "EphA3"
        chem = (echem_ephA3(xs, ys, s, alpha, beta, xret, yret, epha3_mask, epha3_level)) ./ length(xs)
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, sigmaret, gamma), dims=2) ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, s, delta), dims=2) ./ length(xs)^2
    elseif case == "Math5"
        chem = (echemi(xs, ys, s, alpha, beta, xret, yret)) ./ length(xs)
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, sigmaret, gamma), dims=2) ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, s, 0), dims=2) ./ length(xs)^2
    end
    return sum(chem .+ act .+ comp)
end
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Minimisation.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function gpu_optimise(xs, ys, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, eta, hyp1, hyp2, epha3_mask, epha3_level, case)
    # opt = Flux.Optimise.ADAM(eta, (hyp1, hyp2))
    opt = Flux.Optimise.NADAM(eta, (hyp1, hyp2))
    # opt = Flux.Optimise.RADAM(eta, (hyp1, hyp2))
    # opt = Flux.Optimise.RMSProp(eta, hyp1)
    #opt = Flux.Optimise.Nesterov(eta, hyp1)
    #opt = Flux.Optimise.Momentum(eta, hyp1)
    #opt = Flux.Optimise.Descent(eta)
    loss() = energy(xs, ys; s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, epha3_mask, epha3_level, case)
    @showprogress for t = 1:T
        θ = Flux.Params([xs, ys])
        θ̄bar= Flux.gradient(() -> loss(), θ)
        Flux.Optimise.update!(opt, θ, θ̄bar)
        # # enforce boundary conditions
        θ[1][θ[1] .> 1] .= 1.0
        θ[1][θ[1] .< 0] .= 0.0
        θ[2][θ[2] .> 1] .= 1.0
        θ[2][θ[2] .< 0] .= 0.0
    end
    return xs, ys
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Theoretical Timing.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function theoretical_timer(xs, ys, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, eta, hyp1, hyp2, epha3_mask, epha3_level, case)
    for t = 1:T
        energy(xs, ys; s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, epha3_mask, epha3_level, case)
    end
    return nothing
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

function rainbow_plot_kernel(kernel::TopographicKernel, label; pal=0.45, sz1=4, sz2=1.5, DPI = 400)
    x_ret = kernel.kernel[:, 1]
    y_ret = kernel.kernel[:, 2]
    x_col = kernel.kernel[:, 3]
    y_col = kernel.kernel[:, 4]


    cols = map((x, y) -> RGB(x, pal, y), x_ret ./ maximum(x_ret), y_ret ./ maximum(y_ret))

    plt1 = scatter(x_ret, y_ret, markercolor=cols, msc=cols, markersize=sz1, xlim=(0,1), ylim=(0,1), xlabel="Scaled Nasal-Temporal Field", ylabel="Scaled Dorsal-Ventral Field", title="$(label): Pre-Synaptic", legend=false, aspect_ratio=1)
    plt2 = scatter(x_col, y_col, markercolor=cols, msc=cols, markersize=sz2, xlim=(0,1), ylim=(0,1), xlabel="Scaled Rostral-Caudal Axes", ylabel="Scaled Medial-Lateral Axes", title="$(label): Post-Synaptic", legend=false, aspect_ratio=1)
    final = plot(plt1, plt2, layout=(1,2), dpi=DPI)
    return final
end

# end module
end