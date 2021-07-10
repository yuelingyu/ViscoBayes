
function generateparam(ntau::Int64)
    return OrderedDict{Symbol, Any}(
        :log_c0 => 1,
        :log_c => ntau,
        :log_τ => ntau
        )
end

function computetimelimit(t::Vector)
    t[1] == 0 ? time_min = t[2] : time_min = t[1]
    time_max = last(t)
    return time_min, time_max
end

function res_funcmin_s(si, λ, data)

    # λ = exp.(log_λ)
    # si = exp.(log_s)
    
    ntau = length(λ)
    s0 = si[1]
    s = si[2 : (ntau+1)]
    
    dist =  vcat([data[i][:,3] - creepRecursive(s0, s, λ, data[i][:,1], data[i][:,2]) 
            for i in 1:length(data)]...)
            
    return mean(abs2.(dist))
end

function initialparam_s!(startpoint, data::Vector{Matrix{Float64}})
    θ = exp.(startpoint)

    ntau = Int((length(startpoint) - 1)/2)

    s_index = 1 : (ntau + 1)
    λ = θ[(ntau + 2) : (2*ntau + 1)]

    # lower = 1e-9 * ones(ntau + 1)
    # upper = [Inf for _ = 1:(ntau+1)]

    f(si) = res_funcmin_s(si, λ, data)

    res = optimize(f, startpoint[s_index], BFGS())
    # res = optimize(f, lower, upper, startpoint[s_index], BFGS())
    s_opt = Optim.minimizer(res)

    # option 1 
    startpoint[s_index] = log.(abs.(s_opt))
    # option 2
    # startpoint[s_index] = log.(s_opt)

    return startpoint
end


function computetimelimitdata(data_time::Vector{Vector{Float64}})
    ndata = length(data_time)
    t_min = Array{Float64}(undef, ndata)
    t_max = Array{Float64}(undef, ndata)
    for i = 1:ndata
        time_min, time_max = computetimelimit(data_time[i])
        t_min[i] = time_min
        t_max[i] = time_max
    end
    return minimum(t_min), maximum(t_max)
end

function initialtau(time_min, time_max, ntau::Int64)

    u_tau = ceil(log10(time_max)) + 1.0
    l_tau = floor(log10(time_min)) - 1.0

    log_utau = log(10.0^-l_tau)
    log_ltau = log(10.0^-u_tau)

    dist_tau = collect(range(log_ltau, stop = log_utau, length = ntau + 2))

    return dist_tau[2:(end-1)]

end


function initialparam(param::OrderedDict, data::Vector{Matrix{Float64}})
    """
    Generate the initial parameters for mcmc sampler.
    Input:  data -> Vector contains all data
            paramname -> dict of parameters name
    Output: log_param -> Vector of log(param)
    """
    ndata = length(data)
    data_time = [data[i][:, 1] for i in 1:ndata]

    ntau = param[:log_τ]
    startpoint = rand(2*ntau + 1)
    

    τ_index = (ntau + 2) : (2*ntau + 1)
    t_min, t_max = computetimelimitdata(data_time)

    startpoint[τ_index] = initialtau(t_min, t_max, ntau)
    
    initialparam_s!(startpoint, data)
    return startpoint
end

function sim_paramname(name_parms::OrderedDict)
    names_parameters = String[]
    for parm in name_parms
        if parm[2] == 1
            append!(names_parameters, [String(parm[1])])
        else
            names_parm = [String(parm[1])*"[$i]" for i in 1:parm[2]]
            append!(names_parameters, names_parm)
        end
    end
    return names_parameters
end


function read_datalist(data_list)
    ndata = length(data_list)

    data_raw = Array{Array{Float64,2},1}(undef, ndata)

    for i_data in 1:ndata
        df = DataFrame(CSV.File(data_list[i_data]))
        data = Matrix{Float64}(df)
        data = hcat(data, 2*(data[:,3] - data[:,4]))
        data = hcat(data, 3*(data[:,3] + 2.0*data[:,4]))
        data_raw[i_data] = data
    end

    return data_raw
end

function strain_dagger_to_axial_trans(εdagger1, εdagger2)
    ε11 = (1/3) * (εdagger1 + εdagger2/3)
    ε22 = (1/3) * (εdagger2 / 3 - εdagger1/2)
    return ε11, ε22
end


function strain_axial_trans_to_dagger(ε11, ε22)
    εdagger1 = 2*(ε11 - ε22)
    εdagger2 = 3*(ε11 + 2.0*ε22)
    return εdagger1, εdagger2
end
