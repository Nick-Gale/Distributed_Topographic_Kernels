
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
        rs = zeros(N)
        for i =  1:Int(N/nkern)
            ind = (i - 1) * nkern + 1
            rs[ind:(ind+nkern-1)] .= ones(nkern) * rand()
        end
        epha3_mask = map(x -> rs[x] < epha3_fraction, 1:length(xret))
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
        r, th = [rand() * 0.5, rand() * 2 * pi]
        test = true
        x, y = [100, 100]
        while test
            x, y = rand(2) .- 0.5
            test = (x)^2 + (y)^2 > 0.25
        end 
        xret[i:imax] .= x # r * cos(th) + 0.5 # mod(ind, L) / L .+ 1f-7
        yret[i:imax] .= y # r * sin(th) + 0.5 # floor(ind / L) / L .+ 1f-7
    end
    return xret, yret
end

function collicular_initialisation(xret, yret)
    x = 0.0 .* xret .+ 0.01 .* (rand(length(xret)))
    y = 0.0 .* yret .+ 0.01 .* (rand(length(xret)))
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
    # a = alpha .* (1.75 .* exp.(K .* (xi .- 0.5)) .* exp.(K .* (-xret .+ 0.5)) .+ exp.(K .* (xret .- 0.5)) .* exp.(K .* (-xi .+ 0.5))) # alpha .* (erf.((1 .- yi) ./ s) .+ erf.(yi ./ s)) .* (exp.(xret .- xi) .* (erf.((2 .+ s^2 .- 2 .* xi) ./ (2*s)) .+ erf.(s/2 .- xi ./ s)) .+ exp.(xi .- xret) .* (erf.((-2 .+ s^2 .- 2 .* xi) ./ (2*s)) .+ erf.(s/2 .+ xi ./ s))) # 
    # b = beta .* (1.75 .* exp.(K .* (yi .- 0.5)) .* exp.(K .* (-yret .+ 0.5)) .+ exp.(K .* (yret .- 0.5)) .* exp.(K .* (-yi .+ 0.5)))  # beta .* (erf.((1 .- xi) ./ s) .+ erf.(xi ./ s)) .* (exp.(yret .- yi) .* (erf.((2 .+ s^2 .- 2 .* yi) ./ (2*s)) .+ erf.(s/2 .- yi ./ s)) .+ exp.(yi .- yret) .* (erf.((-2 .+ s^2 .- 2 .* yi) ./ (2*s)) .+ erf.(s/2 .+ yi ./ s))) # 

    a = alpha .* (2.0 .* exp.(K .* (xi)) .* exp.(K .* (-xret)) .+ 1.0 .* exp.(K .* (xret)) .* exp.(K .* (-xi)))
    b = beta .* (2.0 * exp.(K .* (yi)) .* exp.(K .* (-yret)) .+ 1.0 .* exp.(K .* (yret)) .* exp.(K .* (-yi)))
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

function echemTi(xi, yi, s, alpha, beta, xret, yret, epha3_mask, epha3_level)
    K = 2
    a = alpha .* (2.0 .* exp.(K .* (xi)) .* exp.(K .* (-xret)) .+ 1.0 .* (exp.(K .* (xret))  .+ epha3_level .* epha3_mask) .* exp.(K .* (-xi)))
    b = beta .* (2.0 * exp.(K .* (yi)) .* exp.(K .* (-yret)) .+ 1.0 .* exp.(K .* (yret)) .* exp.(K .* (-yi)))
    # epha3 = alpha .* epha3_mask .* epha3_level # .* exp.(K .* (-xi)) .* exp.(K .* (xret))

    return  a .+ b # .+  epha3 
end


function eactij(xi, yi, xj, yj, s, xreti, yreti, xretj, yretj, sigmacol, sigmaret, gamma)
    # a = (xj.^2 .+ yj'.^2) ./ s^2 .+ ((xi .- xj').^2 .+ (yi .- yj').^2) ./ sigmacol^2 .+ ((xreti .- xretj').^2 + (yreti .- yretj').^2) ./ sigmaret^2
    # b = (erf.((1 .- xj) ./ (sqrt(2)*s)) .+ erf.((1 .+ xj)' ./ (sqrt(2)*s))) .* (erf.((1 .- yj) ./ (sqrt(2)*s)) .+ erf.((1 .- yj)' ./ (sqrt(2)*s)))
    
    # return gamma/8 * exp.(-1/2 .* a) .* b

    # return gamma/8 * exp.(- 1/2 .* ((xi.-xj').^2 .+ (yi.-yj').^2) .* (1 ./ sigmacol^2  .+ 1 ./ sigmaret^2)) # exp.(-1/2 .* ((xj.^2 .+ yj'.^2) ./ s^2 .+ ((xi .- xj').^2 .+ (yi .- yj').^2) ./ sigmacol^2 .+ ((xreti .- xretj').^2 + (yreti .- yretj').^2) ./ sigmaret^2)) .* ((erf.((1 .- xj) ./ (sqrt(2)*s)) .+ erf.((1 .+ xj)' ./ (sqrt(2)*s))) .* (erf.((1 .- yj) ./ (sqrt(2)*s)) .+ erf.((1 .- yj)' ./ (sqrt(2)*s))))
    return gamma/8 * exp.(- 1/2 .* ((xi.-xj').^2 .+ (yi.-yj').^2 .+ (xreti.-xretj').^2 .+ (yreti.-yretj').^2)  ./ sigmaret^2) # exp.(-1/2 .* ((xj.^2 .+ yj'.^2) ./ s^2 .+ ((xi .- xj').^2 .+ (yi .- yj').^2) ./ sigmacol^2 .+ ((xreti .- xretj').^2 + (yreti .- yretj').^2) ./ sigmaret^2)) .* ((erf.((1 .- xj) ./ (sqrt(2)*s)) .+ erf.((1 .+ xj)' ./ (sqrt(2)*s))) .* (erf.((1 .- yj) ./ (sqrt(2)*s)) .+ erf.((1 .- yj)' ./ (sqrt(2)*s))))

end

function ecompij(xi, yi, xj, yj, s, delta)
    # return delta/8 .* exp.(- 1/2 .* ((xi.-xj').^2 .+ (yi.-yj').^2) ./ s^2) .* (erf.((-2 .+ xi .+ xj') ./ (sqrt(2)*s)) .- erf.((xi .+ xj') ./ (sqrt(2)*s))) .* (erf.((-2 .+ yi .+ yj') ./ (sqrt(2)*s)) .- erf.((yi .+ yj') ./ (sqrt(2)*s)))

    return delta/8 .* exp.(- 1/2 .* ((xi.-xj').^2 .+ (yi.-yj').^2) ./ s^2) #  .* (erf.((-1 .+ xi .+ xj') ./ (sqrt(2)*s)) .- erf.((1 .+ xi .+ xj') ./ (sqrt(2)*s))) .* (erf.((-1 .+ yi .+ yj') ./ (sqrt(2)*s)) .- erf.((1 .+ yi .+ yj') ./ (sqrt(2)*s)))
end

function energy(xs, ys; t, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, epha3_mask, epha3_level, case)
    if case == "WT"
        chem = sum(echemTi(xs, ys, s, alpha, beta, xret, yret, epha3_mask, 0)) # sum((echemi(xs, ys, s, alpha, beta, xret, yret))) # ./ length(xs))
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, sigmaret, gamma)) # ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, s, delta)) # ./ length(xs)^2
    elseif case == "ephrinTKO"
        chem = sum(echemTi(xs, ys, s, alpha * 0, beta, xret, yret, epha3_mask, 0)) # echem_ephrinA2A5(xs, ys, s, alpha, beta, xret, yret) ./ length(xs)
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, sigmaret, gamma)) # ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, s, delta)) # ./ length(xs)^2
    elseif case == "EphA3"
        chem = sum(echemTi(xs, ys, s, alpha, beta, xret, yret, epha3_mask, epha3_level)) # (echem_ephA3(xs, ys, s, alpha, beta, xret, yret, epha3_mask, epha3_level)) ./ length(xs)
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, sigmaret, gamma)) # ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, s, delta)) # ./ length(xs)^2
    elseif case == "Math5"
        chem = sum(echemTi(xs, ys, s, alpha, beta, xret, yret, epha3_mask, 0)) # sum(echemi(xs, ys, s, alpha, beta, xret, yret)) # ./ length(xs)
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, sigmaret, gamma)) # ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, s, delta/15)) # ./ length(xs)^2
    elseif case == "Beta2"
        chem = sum(echemTi(xs, ys, s, alpha, beta, xret, yret, epha3_mask, 0)) # sum(echemi(xs, ys, s, alpha, beta, xret, yret)) # ./ length(xs)
        act = sum(eactij(xs, ys, xs, ys, s, xret, yret, xret, yret, sigmacol, 1.0 * sigmaret, 0.05 * gamma)) # ./ length(xs)^2
        comp = sum(ecompij(xs, ys, xs, ys, 1 * s, 1 * delta)) # ./ length(xs)^2
    end
    if t > 2*T/5
        return  chem .+ comp .+ act
    elseif t > T/5
        return chem .+ comp .+ act
    else
        return chem
    end
end
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Minimisation.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function gpu_optimise(xs, ys, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, eta, hyp1, hyp2, epha3_mask, epha3_level, case)
    opt = Flux.Optimise.ADAM(eta, (hyp1, hyp2))
    # opt = Flux.Optimise.NADAM(eta, (hyp1, hyp2))
    # opt = Flux.Optimise.RADAM(eta, (hyp1, hyp2))
    # opt = Flux.Optimise.RMSProp(eta, hyp1)
    # opt = Flux.Optimise.Nesterov(eta, hyp1)
    # opt = Flux.Optimise.Momentum(eta, hyp1)
    # opt = Flux.Optimise.Descent(eta)
    loss(t) = energy(xs, ys; t, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, epha3_mask, epha3_level, case)
    @showprogress for t = 1:T
        θ = Flux.Params([xs, ys])
        θ̄bar= Flux.gradient(() -> loss(t), θ)
        Flux.Optimise.update!(opt, θ, θ̄bar)
        # # enforce boundary conditions
        mask = (θ[1].-0.0).^2 +  (θ[2].-0.0).^2  .> 0.25
        mag = sqrt.(((θ[1].-0.0).^2 +  (θ[2].-0.0).^2))
        θ[1][mask] .= 0.45 .* θ[1][mask] ./ mag[mask]
        θ[2][mask] .= 0.45 .* θ[2][mask] ./ mag[mask]

        # θ[1][θ[1] .> 1] .= 1.0
        # θ[1][θ[1] .< 0] .= 0.0
        # θ[2][θ[2] .> 1] .= 1.0
        # θ[2][θ[2] .< 0] .= 0.0
    end
    return xs, ys
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Theoretical Timing.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function theoretical_timer(xs, ys, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, eta, hyp1, hyp2, epha3_mask, epha3_level, case)
    for t = 1:T
        energy(xs, ys; t, T, s, alpha, beta, gamma, delta, sigmacol, sigmaret, xret, yret, epha3_mask, epha3_level, case)
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

    inj_inds_nasal = (x_ret .+ 0.2) .^2 .+ (y_ret .+ 0.1) .^2 .< 0.002
    inj_inds_cent = (x_ret .- 0.0) .^2 .+ (y_ret .- 0.0) .^2 .< 0.002
    inj_inds_temporal = (x_ret .+ 0.1) .^2 .+ (y_ret .+ 0.0) .^2 .< 0.002

    cols = map((x, y) -> RGBA(x, pal, y, 0.5), (x_ret .+ 0.5) ./ maximum(x_ret .+ 0.5), (y_ret .+ 0.5) ./ maximum(y_ret .+ 0.5))

    plt1 = scatter(x_ret, y_ret, markercolor=cols, msc=cols, markersize=sz1, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), xlabel="Scaled Nasal-Temporal Field", ylabel="Scaled Dorsal-Ventral Field", title="$(label): Pre-Synaptic", legend=false, aspect_ratio=1)
    plt2 = scatter(x_col, y_col, markercolor=cols, msc=cols, markersize=sz2, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), xlabel="Scaled Rostral-Caudal Axes", ylabel="Scaled Medial-Lateral Axes", title="$(label): Post-Synaptic", legend=false, aspect_ratio=1)
    
    # do a nasal injection
    plot!(plt1, x_ret[inj_inds_nasal], y_ret[inj_inds_nasal], color=:red, markersize=sz1, st=:scatter)
    plot!(plt2, x_col[inj_inds_nasal], y_col[inj_inds_nasal], color=:red, markersize=sz1, st=:scatter)

    # # do a nasal injection
    # plot!(plt1, x_ret[inj_inds_cent], y_ret[inj_inds_cent], color=:blue, markersize=sz1, st=:scatter)
    # plot!(plt2, x_col[inj_inds_cent], y_col[inj_inds_cent], color=:blue,  markersize=sz1, st=:scatter)
    
    # # do a nasal injection
    # plot!(plt1, x_ret[inj_inds_temporal], y_ret[inj_inds_temporal], color=:cyan, markersize=sz1, st=:scatter)
    # plot!(plt2, x_col[inj_inds_temporal], y_col[inj_inds_temporal], color=:cyan, markersize=sz1, st=:scatter)

    
    final = plot(plt1, plt2, layout=(1,2), dpi=DPI)
    return final
end

# end module
end