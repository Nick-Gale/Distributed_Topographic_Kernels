# load packages
using Revise
includet("DistributedTopographicKernels.jl")
includet("LatticeMethod.jl")
using .DistributedTopographicKernels
using .LatticeMethod
using Plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set parameters.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

alpha = 1e-1
beta = 1e-1
gamma = -1e0
delta = 1e1

sigmacol = 0.05
sigmaret = 0.05

epha3 = 4e-1

epha3_fraction = 0.5

s = 0.05
nkerns = 4
N = (25)^2 * nkerns
ncontacts = 15
T = 50
et = 0.025
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to generate lattice object from kernel
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function kernel_lattice(kernel::TopographicKernel; collicular_divider=2.0, direction="L", prespecified_inds=nothing)

        # ALLOW TO SPECIFY THE RADIUS
        if prespecified_inds != nothing
            inds = prespecified_inds
        else
            inds = 1:size(kernel.kernel)[1]
        end

        if direction == "L"
            inds_selected = findall(x -> x < collicular_divider, kernel.kernel[inds, 3])
        elseif direction == "R"
            inds_selected = findall(x -> x > collicular_divider, kernel.kernel[inds, 3])
        end
        
        x_ret = kernel.kernel[inds, 1][inds_selected]
        y_ret = kernel.kernel[inds, 2][inds_selected]
        x_col = kernel.kernel[inds, 3][inds_selected]
        y_col = kernel.kernel[inds, 4][inds_selected]

        params_linking = Dict("linking_key" => "phase_linking", "params" => Dict(:phase_parameter => nothing))
        params_lattice = Dict(  "lattice_forward_preimage" => Dict(:intial_points=>200, :spacing_upper_bound=>2.32, :spacing_lower_bound=>1.68, :minimum_spacing_fraction=>0.75, :spacing_reduction_factor=>0.95), 
                                "lattice_reverse_preimage" => Dict(:intial_points=>200, :spacing_upper_bound=>2.32, :spacing_lower_bound=>1.68, :minimum_spacing_fraction=>0.75, :spacing_reduction_factor=>0.95),
                                "lattice_forward_image" => Dict(:radius=>0.1),
                                "lattice_reverse_image" => Dict(:radius=>0.2)
        )
        return TopographicLattice(x_ret, y_ret, x_col, y_col, params_linking, params_lattice)
    end    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plots for each of the phenotpyes
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# trial_cases = ["WT", "EphA3", "ephrinA2A5", "Math5"] # 
# for current_case in trial_cases
#     tk = TopographicKernel(N, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; eta=et, case=current_case, epha3_level=epha3, epha3_fraction=epha3_fraction)
#     println([tk.rtime, tk.ttime])
#     # plot the raw data
#     plt = rainbow_plot_kernel(tk, "$(current_case)"; sz2=3, pal=0.45)
#     savefig(plt, "figure_distributed_kernels_$(current_case).png")

#     # plot the lattice plots

#     lo = kernel_lattice(tk; direction = "L", collicular_divider = 2.0)
#     p = lattice_plot(lo)
#     implot = plot(p[1], p[4], p[2], p[3], 
#                 title = ["$(current_case): Forward" "$(current_case): Reverse" "" ""],
#                 xlabel = ["Naso-temporal" "Naso-temporal" "Rostro-caudal" "Rostro-caudal"],
#                 ylabel = ["Dorsal-ventral" "Dorsal-ventral" "Medial-Lateral" "Medial-Lateral"],
#                 layout = (2,2), dpi=500)
#     savefig(implot, "figure_lattice_$(current_case).png")
# end

# # do the lattice plots for the EphA3 case seperately
#     tk_epha3 = TopographicKernel(N, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; case="EphA3", epha3_level=epha3, epha3_fraction=epha3_fraction)
    
#     epha3_inds = findall(x -> x == 1, tk_epha3.kernel[:,5])
#     wt_inds = findall(x -> x == 0, tk_epha3.kernel[:,5])

#     lo_wt_projection = kernel_lattice(tk_epha3; prespecified_inds=wt_inds)
#     p_wt = lattice_plot(lo_wt_projection)
    
#     lo_epha3_projection = kernel_lattice(tk_epha3; prespecified_inds=epha3_inds)
#     p_epha3 = lattice_plot(lo_epha3_projection)

#     lo_rostral = kernel_lattice(tk_epha3; collicular_divider=0.4, direction="L")
#     p_rostral = lattice_plot(lo_rostral)

#     lo_caudal = kernel_lattice(tk_epha3; collicular_divider=0.4, direction="R")
#     p_caudal = lattice_plot(lo_caudal)

#     pre_implot = plot(p_wt[1], p_epha3[1], p_wt[2], p_epha3[2], title = ["EphA3-WT: Forward" "EphA3-Ilset2: Forward" "" ""],
#                 xlabel = ["Naso-temporal" "Naso-temporal" "Rostro-caudal" "Rostro-caudal"],
#                 ylabel = ["Dorsal-ventral" "Dorsal-ventral" "Medial-Lateral" "Medial-Lateral"],
#                 layout = (2,2), dpi=500)
#     savefig(pre_implot, "figure_lattice_EphA3_pre.png")

#     post_implot = plot(p_rostral[4], p_caudal[4], p_rostral[3], p_caudal[3], title = ["EphA3-Rostral: Reverse" "EphA3-Caudal: Reverse" "" ""], 
#                 xlabel = ["Naso-temporal" "Naso-temporal" "Rostro-caudal" "Rostro-caudal"],
#                 ylabel = ["Dorsal-ventral" "Dorsal-ventral" "Medial-Lateral" "Medial-Lateral"],
#                 layout = (2,2), dpi=500)
#     savefig(post_implot, "figure_lattice_EphA3_post.png")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Runtime plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    using GLM
    using DataFrames

    koulakov_data = [1000, 25], [2000, 275], [3000, 973], [4000, 2370] # [[100, 1.9], [500, 4.1], [1000, 25], [2000, 275], [3000, 973], [4000, 2370]]#, [5000, 4755]]
    distributed_kernels_data = []
    distributed_kernels_theoretical = []

    for i in koulakov_data
        Ni = round(Int, i[1])
        tk = TopographicKernel(Ni, 1, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; case=trial_cases[1], epha3_level=epha3, epha3_fraction=epha3_fraction)
        push!(distributed_kernels_data, [Ni, tk.rtime])
        push!(distributed_kernels_theoretical, [Ni, tk.ttime])
    end

    koulakov_x = [log(i[1]) for i in koulakov_data]
    koulakov_t = [log(i[2]) for i in koulakov_data]

    d_x = [log(i[1]) for i in distributed_kernels_data]
    d_t = [log(i[2]) for i in distributed_kernels_data]

    t_x = [log(i[1]) for i in distributed_kernels_data]
    t_t = [log(i[2]) for i in distributed_kernels_theoretical]
    

    koulakov_fit = lm(@formula(T ~ X), DataFrame(X=koulakov_x, T=koulakov_t))
    d_fit = lm(@formula(T ~ X), DataFrame(X=d_x, T=d_t))
    t_fit = lm(@formula(T ~ X), DataFrame(X=t_x, T=t_t))

    runtime_plt = plot()
    plot!(runtime_plt, koulakov_x, koulakov_t, seriestype=:line, markershape=:rect, color=RGB(0, 0.4470, 0.7410), label="Tsiganov-Koulakov")
    #plot!(runtime_plt, koulakov_x, koulakov_t, seriestype=:scatter, color=RGB(0, 0.4470, 0.7410))

    plot!(runtime_plt, d_x, d_t, seriestype=:line, markershape=:rect, color=RGB(0.6350, 0.0780, 0.1840), label="Distributed Kernels Method")
    #plot!(runtime_plt, d_x, d_t, seriestype=:scatter, color=RGB(0.6350, 0.0780, 0.1840), label="False")

    plot!(runtime_plt, t_x, t_t, seriestype=:line, markershape=:circle, linestyle=:dash, alpha=0.2, color=RGB(0.0, 0.0, 0.0), label="Theoretical Limit")

    annotate!(runtime_plt, 7.12, 7, text("Tsiganov-Koulakov Fit: \n $(round(coef(koulakov_fit)[2], digits=2))x +$(round(coef(koulakov_fit)[1], digits=2)) \n R² = $(round(r2(koulakov_fit), digits=2))", :black, 10))
    annotate!(runtime_plt, 7.12, 5, text("Distributed Kernels Fit: \n $(round(coef(d_fit)[2], digits=2))x +$(round(coef(d_fit)[1], digits=2)) \n R² = $(round(r2(d_fit), digits=2))", :black, 10))

    plot!(runtime_plt, title = "Wall-Clock Runtime Comparison", xlabel = "log(N)", ylabel = "log(t)", dpi=500)
    savefig(runtime_plt, "figure_runtime.png")
