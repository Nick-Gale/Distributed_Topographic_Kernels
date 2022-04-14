module LatticeMethod

using QHull
using StatsBase
using MiniQhull
using Plots
using Random
using GeometryTypes
using ProgressMeter


export test_lattice
export topograph_linking
export TopographicLattice
export exist_line_intersection
export lattice_plot
export kernel_lattice
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Type Definitions
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct TopographicLattice
    # the raw pre_synaptic and post_synaptic co-ordinate locations
    pre_synaptic::Array{Float64, 2}
    post_synaptic::Array{Float64, 2}

    # the data defined topographic map under some measurement
    topographic_map::Array{Float64, 2}

    # the forward projection 
    forward_preimage::Array{Float64, 2}
    forward_image::Array{Float64, 2}
    forward_links_removed::Array{Int, 2}
    forward_links_retained::Array{Int, 2}

    # the reverse projection 
    reverse_preimage::Array{Float64, 2}
    reverse_image::Array{Float64, 2}
    reverse_links_removed::Array{Any, 2}
    reverse_links_retained::Array{Int, 2}

    # the construction method 
    function TopographicLattice(pre_x::Array{Float64, 1}, pre_y::Array{Float64, 1}, post_x::Array{Float64, 1}, post_y::Array{Float64, 1}, params_linking::Any, params_lattice::Any)
        
        # construct the topographic map by some linking function 
        pre_synaptic = hcat(pre_x, pre_y)
        post_synaptic = hcat(post_x, post_y)
        @time topographic_map, pre_synaptic, post_synaptic = topographic_linking(pre_synaptic, post_synaptic, params_linking)
    
        # define the projections pre_image on a restricted number of points
        println("Selecting Points")
        @time forward_preimage_points = select_projection_points(pre_synaptic; params_lattice["lattice_forward_preimage"]...)
        @time reverse_preimage_points = select_projection_points(post_synaptic; params_lattice["lattice_reverse_preimage"]...)
        forward_preimage = pre_synaptic[forward_preimage_points, :]
        reverse_preimage = post_synaptic[reverse_preimage_points, :]
        
        println("Creating Images")
        # create the images 
        @time forward_image = create_projection(forward_preimage_points, transpose(topographic_map), pre_synaptic, post_synaptic; params_lattice["lattice_forward_image"]...)
        @time reverse_image = create_projection(reverse_preimage_points, topographic_map, post_synaptic, pre_synaptic; params_lattice["lattice_reverse_image"]...)

        # create the graph adjacencies based on a delaunay triangulation; graph adjacencies indexed on 1:length(projection)
        forward_adjacency_sparse, forward_triangulation  = adj_mat(forward_preimage)
        reverse_adjacency_sparse, forward_triangulation = adj_mat(reverse_preimage)

        # remove any overlapping links when the functional map is applied to the graph, the links remaining define the lattice object
        forward_links_retained, forward_links_removed = link_crossings(forward_adjacency_sparse, forward_image)
        reverse_links_retained, reverse_links_removed = link_crossings(reverse_adjacency_sparse, reverse_image)

        new(pre_synaptic, post_synaptic, topographic_map, forward_preimage, forward_image, forward_links_removed, forward_links_retained,
        reverse_preimage, reverse_image, reverse_links_removed, reverse_links_retained)
    end
end

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Projection Functions
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

function select_projection_points(coordinates; intial_points=200, spacing_upper_bound=2.32, spacing_lower_bound=1.68, minimum_spacing_fraction=0.75, spacing_reduction_factor=0.95)
    area = abs(chull(coordinates).area)
    mean_spacing = 0;
    n_points = intial_points
    points_selected = []
    Random.seed!(1)
    # While the spacing is not in bounds we target a minimum spacing and then randomly select n_points which are good candidates for the desired target spacing. 
    # When we select the points we want to ensure that there are a number of points around it within a given radius so that the projection has a good quality. This step is ommitted in this version.

    while ((mean_spacing < spacing_lower_bound) || (mean_spacing > spacing_upper_bound)) && (n_points > 1)
        # set the spacing
        min_spacing = minimum_spacing_fraction * sqrt(area / n_points)
        while length(points_selected) < n_points
            # if a good trial set was not found reset the selected points and potential points and try again after reducing the spacing
            potential_points = collect(1:size(coordinates)[1])
            points_selected = []

            # if the required number of points have not yet been chosen and there are still some points to be chosen from then choose a new point
            while (length(points_selected) < n_points) && (length(potential_points)>0)
                # chose a candidate point, add it to the selected points list and remove it from potential further selections
                candidate = rand(1:length(potential_points))
                selected_index = potential_points[candidate]
                append!(points_selected, selected_index)
                deleteat!(potential_points, candidate)

                # now remove all potential points within the mininum spacing of this chosen point and remove them
                unacceptable_indexes = findall(x -> sqrt((coordinates[x, 1] - coordinates[selected_index, 1])^2 + (coordinates[x, 2] - coordinates[selected_index, 2])^2) < min_spacing, 1:size(coordinates)[1])
                potential_points = setdiff(potential_points, unacceptable_indexes)
            end

            # reduce the spacing
            min_spacing *= spacing_reduction_factor
        end

        #compute the spacings to check if within bounds and reduce the number of points if necessary to continue the loop
        distances = sqrt.((coordinates[points_selected, 1] .- coordinates[points_selected, 1]') .^ 2 .+ (coordinates[points_selected, 2] .- coordinates[points_selected, 2]') .^ 2)
        mean_spacing = mean(distances .+ maximum(distances))
        n_points -= 1
    end
    
    return points_selected
end

function create_projection(preimage_points, adjacency, preimage_coordinates, image_coordinates; radius=0.01)
    projected_image = zeros(length(preimage_points), 2)
    # be careful with your adjacency matrix in this function - it might need to be transposed
    @showprogress for i = 1:length(preimage_points)
        pre_image_projected_radius = findall(x -> sqrt((preimage_coordinates[x, 1] - preimage_coordinates[preimage_points[i], 1])^2 + (preimage_coordinates[x, 2] - preimage_coordinates[preimage_points[i], 2])^2) < radius, 1:size(preimage_coordinates)[1])
        image_projected_points = []
        for j in pre_image_projected_radius
            push!(image_projected_points, findall(adjacency[j, :].>0)...)
        end
        projected_image[i, 1] = mean(image_coordinates[image_projected_points, 1])
        projected_image[i, 2] = mean(image_coordinates[image_projected_points, 2])
    end
    return projected_image 
end

function adj_mat(points)
    triangulation_abstract = delaunay(points')
    adjacency = zeros(Int64, size(triangulation_abstract)[2] * 3, 2)
    for i = 1:size(triangulation_abstract)[2]
        p = triangulation_abstract[1, i]
        q = triangulation_abstract[2, i]
        r = triangulation_abstract[3, i]
        adjacency[3 * (i - 1) + 1, :] = [p,q]
        adjacency[3 * (i - 1) + 2, :] = [r,q]
        adjacency[3 * (i - 1) + 3, :] = [p,r]
    end
    return adjacency, triangulation_abstract
end

function link_crossings(adjacency_sparse, projection)
    indexes_links_removed = []
    for i = 1:size(adjacency_sparse)[1]
        for j = (i + 1):size(adjacency_sparse)[1]
            if exist_line_intersection(adjacency_sparse[i, :], adjacency_sparse[j, :], projection)
                push!(indexes_links_removed, i)
                push!(indexes_links_removed, j)
            end
        end
    end
    indexes_links_removed = unique(indexes_links_removed)
    indexes_links_remain = setdiff(collect(1:size(adjacency_sparse)[1]), indexes_links_removed)
    return adjacency_sparse[indexes_links_remain,:], adjacency_sparse[indexes_links_removed,:]
end

function ccw(p1, p2, p3)
    return (p3[2] - p1[2]) * (p2[1] - p1[1]) > (p2[2] - p1[2]) * (p3[1] - p1[1])
end

function exist_line_intersection(e1, e2, projection)
    p1 = projection[e1[1], :]; p2 = projection[e1[2], :]; p3 = projection[e2[1], :]; p4 = projection[e2[2], :];
    tv1 = ccw(p1, p3, p4) != ccw(p2, p3, p4)
    tv2 = ccw(p1, p2, p3) != ccw(p1, p2, p4)
    touching_ends = any([all(p1 .== p3), all(p1 .== p4), all(p2 .== p3), all(p2 .== p4)])
    return tv1 * tv2 * (!touching_ends)
end

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Statistics Functions
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
function map_quality(lattice::TopographicLattice)
    return length(reverse_links_retained) / (length(reverse_links_retained) + length(reverse_links_removed))
end

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting Functions
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
function lattice_plot(lattice::TopographicLattice; print_removed_links=true)
    # this needs to be added with parameters
    
    #outline
    par_t = vcat(collect(1:1000), [1]) ./ 1000 .* 2 .* pi
    circ_x = 0.5 .* cos.(par_t)
    circ_y = 0.5 .* sin.(par_t)

    # data
    forward_preimage = lattice.forward_preimage
    forward_image = lattice.forward_image
    forward_links = lattice.forward_links_retained
    forward_remove = lattice.forward_links_removed

    reverse_preimage = lattice.reverse_preimage
    reverse_image = lattice.reverse_image
    reverse_links = lattice.reverse_links_retained
    reverse_remove = lattice.reverse_links_removed
    
    # forward preimage
        forward_preimage_plot = plot()
        for i = 1:size(forward_links)[1]
            # each link input in the form [x1, y1], [x2, y2]
            e1 = forward_links[i, 1]
            e2 = forward_links[i, 2]
            x1 = forward_preimage[e1, 1]
            y1 = forward_preimage[e1, 2]
            x2 = forward_preimage[e2, 1]
            y2 = forward_preimage[e2, 2]
            plot!(forward_preimage_plot, [x1, x2], [y1, y2], legend=false, color=:blue, aspect_ratio=:equal)
            plot!(forward_preimage_plot, circ_x, circ_y, seriestype=:path, color=:purple, style=:dash, width=1.5)
        end

    # forward image
        forward_image_plot = plot()
        for i = 1:size(forward_links)[1]
            # each link input in the form [x1, y1], [x2, y2]
            e1 = forward_links[i, 1]
            e2 = forward_links[i, 2]
            x1 = forward_image[e1, 1]
            y1 = forward_image[e1, 2]
            x2 = forward_image[e2, 1]
            y2 = forward_image[e2, 2]
            plot!(forward_image_plot, [x1, x2], [y1, y2], legend=false, color=:blue, aspect_ratio=:equal)
            plot!(forward_image_plot, circ_x, circ_y, seriestype=:path, color=:purple, style=:dash, width=1.5)
        end

    # reverse preimage
        reverse_preimage_plot = plot()
        for i = 1:size(reverse_links)[1]
            # each link input in the form [x1, y1], [x2, y2]
            e1 = reverse_links[i, 1]
            e2 = reverse_links[i, 2]
            x1 = reverse_preimage[e1, 1]
            y1 = reverse_preimage[e1, 2]
            x2 = reverse_preimage[e2, 1]
            y2 = reverse_preimage[e2, 2]
            plot!(reverse_preimage_plot, [x1, x2], [y1, y2], legend=false, color=:black, aspect_ratio=:equal)
            plot!(reverse_preimage_plot, circ_x, circ_y, seriestype=:path, color=:purple, style=:dash, width=1.5)
        end

    # reverse image
        reverse_image_plot = plot()
        for i = 1:size(reverse_links)[1]
            # each link input in the form [x1, y1], [x2, y2]
            e1 = reverse_links[i, 1]
            e2 = reverse_links[i, 2]
            x1 = reverse_image[e1, 1]
            y1 = reverse_image[e1, 2]
            x2 = reverse_image[e2, 1]
            y2 = reverse_image[e2, 2]
            plot!(reverse_image_plot, [x1, x2], [y1, y2], legend=false, color=:black, aspect_ratio=:equal)
            plot!(reverse_image_plot, circ_x, circ_y, seriestype=:path, color=:purple, style=:dash, width=1.5)
        end


    if print_removed_links
        for i = 1:size(reverse_remove)[1]
            # each link input in the form [x1, y1], [x2, y2]
            e1 = reverse_remove[i, 1]
            e2 = reverse_remove[i, 2]
            x1 = reverse_image[e1, 1]
            y1 = reverse_image[e1, 2]
            x2 = reverse_image[e2, 1]
            y2 = reverse_image[e2, 2]
            plot!(reverse_image_plot, [x1, x2], [y1, y2], legend=false, color=:red, aspect_ratio=:equal)
        end

        for i = 1:size(reverse_remove)[1]
            # each link input in the form [x1, y1], [x2, y2]
            e1 = reverse_remove[i, 1]
            e2 = reverse_remove[i, 2]
            x1 = reverse_preimage[e1, 1]
            y1 = reverse_preimage[e1, 2]
            x2 = reverse_preimage[e2, 1]
            y2 = reverse_preimage[e2, 2]
            plot!(reverse_preimage_plot, [x1, x2], [y1, y2], legend=false, color=:red, aspect_ratio=:equal)
        end

        for i = 1:size(forward_remove)[1]
            # each link input in the form [x1, y1], [x2, y2]
            e1 = forward_remove[i, 1]
            e2 = forward_remove[i, 2]
            x1 = forward_image[e1, 1]
            y1 = forward_image[e1, 2]
            x2 = forward_image[e2, 1]
            y2 = forward_image[e2, 2]
            plot!(forward_image_plot, [x1, x2], [y1, y2], legend=false, color=:red, aspect_ratio=:equal)
        end

        for i = 1:size(forward_remove)[1]
            # each link input in the form [x1, y1], [x2, y2]
            e1 = forward_remove[i, 1]
            e2 = forward_remove[i, 2]
            x1 = forward_preimage[e1, 1]
            y1 = forward_preimage[e1, 2]
            x2 = forward_preimage[e2, 1]
            y2 = forward_preimage[e2, 2]
            plot!(forward_preimage_plot, [x1, x2], [y1, y2], legend=false, color=:red, aspect_ratio=:equal)
        end
    end
    xl = (-0.5, 0.5)
    plot!(forward_preimage_plot, forward_preimage[:, 1], forward_preimage[:, 2], xlim=xl, ylim=xl, seriestype=:scatter, color=:blue)
    plot!(reverse_preimage_plot, reverse_preimage[:, 1], reverse_preimage[:, 2], xlim=xl, ylim=xl, seriestype=:scatter, color=:black)
    plot!(reverse_image_plot, reverse_image[:, 1], reverse_image[:, 2], xlim=xl, ylim=xl, seriestype=:scatter, color=:black)
    plot!(forward_image_plot, forward_image[:, 1], forward_image[:, 2], xlim=xl, ylim=xl, seriestype=:scatter, color=:blue)

    return forward_preimage_plot, forward_image_plot, reverse_preimage_plot, reverse_image_plot
end

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Testing Functions
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
function test_lattice(n::Int64, s::Int64)
    Random.seed!(s)
    pre_x = rand(n); pre_y = rand(n); post_x = pre_x .+ 0.01 * rand(n); post_y = pre_y .+ 0.01 * rand(n);
    params_linking = Dict("linking_key" => "asscociation", "params" => Dict(:radius=>0.1))
    params_lattice = Dict("lattice_forward_preimage" => Dict(:intial_points=>round(Int64, n/5), :spacing_upper_bound=>2.32, :spacing_lower_bound=>1.68, :minimum_spacing_fraction=>0.75, :spacing_reduction_factor=>0.95), 
                           "lattice_reverse_preimage" => Dict(:intial_points=>round(Int64, n/5), :spacing_upper_bound=>2.32, :spacing_lower_bound=>1.68, :minimum_spacing_fraction=>0.75, :spacing_reduction_factor=>0.95),
                           "lattice_forward_image" => Dict(:radius=>0.05),
                           "lattice_reverse_image" => Dict(:radius=>0.05)
                           )
    @time lattice_object = TopographicLattice(pre_x, pre_y, post_x, post_y, params_linking, params_lattice)
    p1, p2, p3, p4 = lattice_plot(lattice_object)
    plt = plot(p1, p2, p3, p4, layout=(2,2), dpi=500)
    savefig(plt, "plot.png")
    return nothing
end

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Topographic Map Construction From Data
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


function topographic_linking(pre_synaptic, post_synaptic, params_linking)
    linking_key = params_linking["linking_key"]
    if linking_key == "phase_linking"
        return topographic_phase_linking(pre_synaptic, post_synaptic; params_linking["params"]...)
    end

    if linking_key == "asscociation"
        return topographic_asscociation(pre_synaptic, post_synaptic; params_linking["params"]...)
    end
end

function topographic_phase_linking(pre_synaptic, post_synaptic; phase_parameter=0.0)
    pre_list = [[]]
    for i = 1:size(pre_synaptic)[1]
        push!(pre_list, pre_synaptic[i, :])
    end
    deleteat!(pre_list, 1)
    pre_list = unique(pre_list)

    array = zeros(Int64, length(pre_list), size(post_synaptic)[1])
    for j = 1:length(pre_list)
        connected_inds = getindex.(findall(x -> all(pre_synaptic[x, :] .== pre_list[j]), 1:size(pre_synaptic)[1]), 1)
        for i in connected_inds
            array[j, i] += 1
        end
    end
    new_presynaptic = zeros(Float64, length(pre_list), 2)
    for i = 1:length(pre_list)
        new_presynaptic[i, 1] = pre_list[i][1]
        new_presynaptic[i, 2] = pre_list[i][2]
    end
    
    println(size(array))
    return transpose(array), new_presynaptic, post_synaptic
end

function topographic_asscociation(pre_synaptic, post_synaptic; radius=0.0001)
    array = zeros(Int64, size(post_synaptic)[1], size(pre_synaptic)[1])
    # pre-synaptic indexes are on columns
    for i = 1:size(array)[2]
        connected_inds = getindex.(findall(x -> sqrt((pre_synaptic[i, 1] - post_synaptic[x, 1])^2 + (pre_synaptic[i, 2] - post_synaptic[x, 2])^2) < radius, 1:size(post_synaptic)[1]), 1)
        for j in connected_inds
            array[j, i] += 1
        end
    end
    return array
end

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to generate lattice object from kernel
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function kernel_lattice(kernel; points=250, collicular_divider=2.0, direction="L", prespecified_inds=nothing, r1=0.05, r2=0.05)

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
    params_lattice = Dict(  "lattice_forward_preimage" => Dict(:intial_points=>points, :spacing_upper_bound=>2.32, :spacing_lower_bound=>1.68, :minimum_spacing_fraction=>0.75, :spacing_reduction_factor=>0.95), 
                            "lattice_reverse_preimage" => Dict(:intial_points=>points, :spacing_upper_bound=>2.32, :spacing_lower_bound=>1.68, :minimum_spacing_fraction=>0.75, :spacing_reduction_factor=>0.95),
                            "lattice_forward_image" => Dict(:radius=>r1),
                            "lattice_reverse_image" => Dict(:radius=>r2)
    )
    return TopographicLattice(x_ret, y_ret, x_col, y_col, params_linking, params_lattice)
end    


# end module
end
