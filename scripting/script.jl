using .DistributedTopographicKernels, .LatticeMethod, Plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set parameters.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

alpha = 1e-0
beta = 1e-0
gamma = -5e-1
delta = 3e-1

sigmacol = 0.05
sigmaret = 0.1

epha3 = 5
epha3_fraction = 0.5

s = 0.075
nkerns = 3
N = (30)^2 * nkerns
ncontacts = 10
T = 200
et = 0.005

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# B2-WT measure comparison
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
trial_cases = ["WT", "Beta2"] 
diameters = []
for current_case in trial_cases
    tk = TopographicKernelTemporal(N, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; eta=et, case=current_case, epha3_level=epha3, epha3_fraction=epha3_fraction)
    push!(diameters, tk.diameter)
end

b2comp = plot(1:length(diameters[1]), diameters, label = ["WT" "Beta2-/-" ""], title="Ellipse Area Covering Projective Field", dpi=500, xlabel="Iterations")
vline!([round(Int, 1/5 * T)], label = "Retinal Activity Onset")
savefig(b2comp, "./figures/figure_beta2vsWTprojection.png")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plots for each of the phenotpyes
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
trial_cases = Dict("WT" => "WT", "EphA3" => "EphA3 Knock-In", "ephrinTKO" => "ephrinTKO", "Math5" => "Math5-/-", "Beta2" => "Beta2-/-") #    
#trial_cases = ["WT", "EphA3", "ephrinTKO", "Math5", "Beta2"]
for (key, current_case) in trial_cases
    tk = TopographicKernel(N, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; eta=et, case=key, epha3_level=epha3, epha3_fraction=epha3_fraction)
    # plot the raw data
    plt = rainbow_plot_kernel(tk, "$(current_case)"; sz2=3, pal=0.45)
    savefig(plt, "./figures/figure_distributed_kernels_$(key).png")

    # plot the lattice plots

    lo = kernel_lattice(tk; direction = "L", collicular_divider = 2.0)
    p = lattice_plot(lo)
    implot = plot(p[1], p[4], p[2], p[3], 
                title = ["$(current_case): Forward" "$(current_case): Reverse" "" ""],
                xlabel = ["Naso-Temporal" "Naso-Temporal" "Rostro-Caudal" "Rostro-Caudal"],
                ylabel = ["Dorsal-Ventral" "Dorsal-Ventral" "Medial-Lateral" "Medial-Lateral"],
                layout = (2,2), dpi=500)
    savefig(implot, "./figures/figure_lattice_$(key).png")
end

# do the lattice plots for the EphA3 case seperately
    tk_epha3 = TopographicKernel(N, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; case="EphA3", epha3_level=epha3, epha3_fraction=epha3_fraction)
    
    epha3_inds = findall(x -> x == 1, tk_epha3.kernel[:,5])
    wt_inds = findall(x -> x == 0, tk_epha3.kernel[:,5])

    lo_wt_projection = kernel_lattice(tk_epha3; prespecified_inds=wt_inds, r1=0.25)
    p_wt = lattice_plot(lo_wt_projection)
    
    lo_epha3_projection = kernel_lattice(tk_epha3; prespecified_inds=epha3_inds, r1=0.25)
    p_epha3 = lattice_plot(lo_epha3_projection)

    lo_rostral = kernel_lattice(tk_epha3; points=150, collicular_divider=0.0, direction="L", r2=0.25)
    p_rostral = lattice_plot(lo_rostral)

    lo_caudal = kernel_lattice(tk_epha3; points=150, collicular_divider=0.0, direction="R", r2=0.25)
    p_caudal = lattice_plot(lo_caudal)

    pre_implot = plot(p_wt[1], p_epha3[1], p_wt[2], p_epha3[2], title = ["EphA3-Islet2(-ve): Forward" "EphA3-Islet2(+ve): Forward" "" ""],
                xlabel = ["Naso-Temporal" "Naso-Temporal" "Rostro-Caudal" "Rostro-Caudal"],
                ylabel = ["Dorsal-Ventral" "Dorsal-Ventral" "Medial-Lateral" "Medial-Lateral"],
                layout = (2,2), dpi=500)
    savefig(pre_implot, "./figures/figure_lattice_EphA3_pre.png")

    post_implot = plot(p_rostral[4], p_caudal[4], p_rostral[3], p_caudal[3], title = ["EphA3-Rostral: Reverse" "EphA3-Caudal: Reverse" "" ""], 
                xlabel = ["Naso-Temporal" "Naso-Temporal" "Rostro-Caudal" "Rostro-Caudal"],
                ylabel = ["Dorsal-Ventral" "Dorsal-Ventral" "Medial-Lateral" "Medial-Lateral"],
                layout = (2,2), dpi=500)
    savefig(post_implot, "./figures/figure_lattice_EphA3_post.png")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Runtime plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    using GLM
    using DataFrames

    koulakov_data = [1000, 25], [2000, 275], [3000, 973], [4000, 2370] # [[100, 1.9], [500, 4.1], [1000, 25], [2000, 275], [3000, 973], [4000, 2370]]#, [5000, 4755]]
    distributed_kernels_data = []

    for i in koulakov_data
        Ni = round(Int, i[1])
        tk = TopographicKernel(Ni, 1, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; case=trial_cases["WT"], epha3_level=epha3, epha3_fraction=epha3_fraction)
        push!(distributed_kernels_data, [Ni, tk.rtime])
    end

    koulakov_x = [log(i[1]) for i in koulakov_data]
    koulakov_t = [log(i[2]) for i in koulakov_data]

    d_x = [log(i[1]) for i in distributed_kernels_data]
    d_t = [log(i[2]) for i in distributed_kernels_data]

    t_x = [log(i[1]) for i in distributed_kernels_data]
    

    koulakov_fit = lm(@formula(T ~ X), DataFrame(X=koulakov_x, T=koulakov_t))
    d_fit = lm(@formula(T ~ X), DataFrame(X=d_x, T=d_t))

    runtime_plt = plot()
    plot!(runtime_plt, koulakov_x, koulakov_t, seriestype=:line, markershape=:rect, color=RGB(0, 0.4470, 0.7410), label="Tsiganov-Koulakov")
    #plot!(runtime_plt, koulakov_fit, label="Tsiganov-Koulakov Fit")

    plot!(runtime_plt, d_x, d_t, seriestype=:line, markershape=:rect, color=RGB(0.6350, 0.0780, 0.1840), label="Distributed Kernels")
    annotate!(runtime_plt, 7.12, 7, text("Tsiganov-Koulakov Fit: \n $(round(coef(koulakov_fit)[2], digits=2))x +$(round(coef(koulakov_fit)[1], digits=2)) \n R² = $(round(r2(koulakov_fit), digits=3))", :black, 10))
    annotate!(runtime_plt, 7.12, 5, text("Distributed Kernels Fit: \n $(round(coef(d_fit)[2], digits=2))x +$(round(coef(d_fit)[1], digits=2)) \n R² = $(round(r2(d_fit), digits=3))", :black, 10))
    #plot!(runtime_plt, d_fit, label="Distributed Kernels Fit")

    plot!(runtime_plt, title = "Wall-Clock Runtime Comparison", xlabel = "log(N)", ylabel = "log(t)", dpi=500)
    savefig(runtime_plt, "./figures/figure_runtime.png")

