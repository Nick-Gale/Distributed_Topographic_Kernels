# load packages
using Revise
includet("DistributedTopographicKernels.jl")
using .DistributedTopographicKernels
using Plots

# set parameters 
alpha = 1e-2
beta = 1e-2
gamma = -1e-0
delta = 1e0

sigmacol = 0.01
sigmaret = 0.01

epha3 = 5e-2

epha3_fraction = 0.4

s = 0.05

nkerns = 5
N = 3000
ncontacts = 50

T = 250

trial_case = "WT"

# initialise object

@time tk = TopographicKernel(N, nkerns, ncontacts, s, alpha, beta, gamma, delta, sigmacol, sigmaret, T; case=trial_case, epha3_level=epha3, epha3_fraction=epha3_fraction)
plt = rainbow_plot_kernel(tk, "Wild Type")
savefig(plt, "figure.png")