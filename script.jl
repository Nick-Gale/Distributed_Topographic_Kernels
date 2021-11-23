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

alpha = 1e-2
beta = 1e-2
gamma = -1e-0
delta = 1e0

sigmacol = 0.01
sigmaret = 0.01

epha3 = 5e-2

epha3_fraction = 0.4

s = 0.05

nkerns = 2
N = 400
ncontacts = 2

T = 50
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to generate lattice object from kernel
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    function kernel_lattice(kernel::TopographicKernel; collicular_divider=2.0, direction="L", prespecified_inds=nothing)
        if direction == "L"
            inds = findall(x -> x < collicular_divider, kernel.kernel[:, 3])
        elseif direction == "R"
            inds = findall(x -> x > collicular_divider, kernel.kernel[:, 3])
        end

        if prespecified_inds != nothing
            inds = prespecified_inds
        end
        
        x_ret = kernel.kernel[inds, 1]
        y_ret = kernel.kernel[inds, 2]
        x_col = kernel.kernel[inds, 3]
        y_col = kernel.kernel[inds, 4]

        params_linking = Dict("linking_key" => "phase_linking", "params" => Dict(:phase_parameter => nothing))
        params_lattice = Dict(  "lattice_forward_preimage" => Dict(:intial_points=>round(Int64, length(x_col)/20), :spacing_upper_bound=>2.32, :spacing_lower_bound=>1.68, :minimum_spacing_fraction=>0.75, :spacing_reduction_factor=>0.95), 
                                "lattice_reverse_preimage" => Dict(:intial_points=>round(Int64, length(x_ret)/20), :spacing_upper_bound=>2.32, :spacing_lower_bound=>1.68, :minimum_spacing_fraction=>0.75, :spacing_reduction_factor=>0.95),
                                "lattice_forward_image" => Dict(:radius=>0.2),
                                "lattice_reverse_image" => Dict(:radius=>0.2)
        )
        return TopographicLattice(x_ret, y_ret, x_col, y_col, params_linking, params_lattice)
    end    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plots for each of the phenotpyes
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
trial_cases = ["WT", "EphA3", "ephrinA2A5", "Math5"]
for current_case in trial_cases
    @time tk = TopographicKernel(N, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; case=current_case, epha3_level=epha3, epha3_fraction=epha3_fraction)
    
    # plot the raw data
    plt = rainbow_plot_kernel(tk, "$(current_case)"; sz2=3)
    savefig(plt, "figure_distributed_kernels_$(current_case).png")

    # plot the lattice plots

    lo = kernel_lattice(tk; direction = "L", collicular_divider = 2.0)
    p = lattice_plot(lo)
    implot = plot(p[1], p[3], p[2], p[4], title = ["$(current_case): Pre-to-Post" "$(current_case): Post-to-Pre" "" ""], layout = (2,2), dpi=500)
    savefig(implot, "figure_lattice_$(current_case).png")
end

# do the lattice plots for the EphA3 case seperately
    tk_epha3 = TopographicKernel(N, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; case="EphA3", epha3_level=epha3, epha3_fraction=epha3_fraction)
    
    epha3_inds = tk_epha3.epha3
    wt_inds = setdiff(1:size(tk_epha3.kernel)[1], epha3_inds)
    
    lo_wt_projection = kernel_lattice(tk_epha3; prespecified_inds=wt_inds)
    p_wt = lattice_plot(lo_wt_projection)
    
    lo_epha3_projection = kernel_lattice(tk_epha3; prespecified_inds=epha3_inds)
    p_epha3 = lattice_plot(lo_epha3_projection)

    lo_rostral = kernel_lattice(tk_epha3; collicular_divider=0.5, direction="L")
    p_rostral = lattice_plot(lo_rostral)

    lo_caudal = kernel_lattice(tk_epha3; collicular_divider=0.5, direction="R")
    p_caudal = lattice_plot(lo_caudal)

    implot = plot(p_wt[1], p_wt[2], p_epha3[1], p_epha3[2], title = ["EphA3-WT: Pre-to-Post" "EphA3-Ilset2: Pre-to-Post" "" ""], layout = (2,2), dpi=500)
    savefig(implot, "figure_lattice_EphA3_pre.png")

    implot = plot(p_rostral[3], p_rostral[4], p_caudal[3], p_caudal[4], title = ["EphA3-Rostral: Post-to-Pre" "EphA3-Caudal: Post-to-Pre" "" ""], layout = (2,2), dpi=500)
    savefig(implot, "figure_lattice_EphA3_post.png")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Runtime plots
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # using GLM
    # using DataFrames

    # koulakov_data = [[100, 0.001], [300, 0.1], [600, 1], [1000, 1], [3000, 1], [20, 1]]
    # distributed_kernels_data = []

    # for i in koulakov_data
    #     Ni = round(Int, i[1])
    #     et = @elapsed TopographicKernel(Ni, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; case=trial_cases[1], epha3_level=epha3, epha3_fraction=epha3_fraction)
    #     push!(distributed_kernels_data, [Ni, et])
    # end

    # koulakov_x = [log(i[1]) for i in koulakov_data]
    # koulakov_t = [log(i[2]) for i in koulakov_data]

    # d_x = [log(i[1]) for i in distributed_kernels_data]
    # d_t = [log(i[2]) for i in distributed_kernels_data]

    # koulakov_fit = lm(@formula(T ~ X), DataFrame(X=koulakov_x, T=koulakov_t))
    # d_fit = lm(@formula(T ~ X), DataFrame(X=d_x, T=d_t))

    # runtime_plt = plot()
    # plot!(runtime_plt, koulakov_x, koulakov_t, seriestype=:line, markershape=:rect, color=RGB(0, 0.4470, 0.7410), label="Tsiganov-Koulakov")
    # #plot!(runtime_plt, koulakov_x, koulakov_t, seriestype=:scatter, color=RGB(0, 0.4470, 0.7410))

    # plot!(runtime_plt, d_x, d_t, seriestype=:line, markershape=:rect, color=RGB(0.6350, 0.0780, 0.1840), label="Distributed Kernels Method")
    # #plot!(runtime_plt, d_x, d_t, seriestype=:scatter, color=RGB(0.6350, 0.0780, 0.1840), label="False")

    # annotate!(runtime_plt, 7, -4, text("Tsiganov-Koulakov Fit: \n $(round(coef(koulakov_fit)[2], digits=2))x +$(round(coef(koulakov_fit)[1], digits=2)) \n R² = $(round(r2(koulakov_fit), digits=2))", :black, 10))
    # annotate!(runtime_plt, 7, -6, text("Distributed Kernels Fit: \n $(round(coef(d_fit)[2], digits=2))x +$(round(coef(d_fit)[1], digits=2)) \n R² = $(round(r2(koulakov_fit), digits=2))", :black, 10))

    # plot!(runtime_plt, title = "Wall-Clock Runtime Comparison", xlabel = "log(N)", ylabel = "log(t)", dpi=500)
    # savefig(runtime_plt, "figure_runtime.png")
