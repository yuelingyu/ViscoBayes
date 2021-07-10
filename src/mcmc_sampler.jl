function viscomcmcsample(ntau, data, info_prior, prior_type, sigma, niter = 1000)

    name_param = generateparam(ntau)
    startpoint = initialparam(name_param, data)

    logf(log_θ) = log_posterior(log_θ, data, info_prior, prior_type)

    # warmup = Int(floor(0.3*niter))
    warmup = 1
    sim = Chains(niter, length(startpoint), start = (warmup + 1), 
                    names = sim_paramname(name_param))

    θ = AMWGVariate(startpoint, sigma, logf)

    @showprogress for i in 1:niter
        sample!(θ, adapt = (i <= warmup))
        if i > warmup
            sim[i, :, 1] = θ
        end
    end

    θ_iter = getiterparam(sim, name_param::OrderedDict)
    θ_MAP, log_post = estimateMAP(logf, θ_iter)

    return θ_iter, θ_MAP, log_post
end


function visco3Dbayes(name_material, data_raw, ntau, prior_type::Vector{Int64}, sigma, niter)

    ndata = length(data_raw)
    
    data_μ = [data_raw[i][:, [1,2,5]] for i in 1:ndata]
    data_κ = [data_raw[i][:, [1,2,6]] for i in 1:ndata]

    mat_info_κ, mat_info_μ = materials_info(name_material)

    println("Starting anlysis of kappa with $ntau series ... ")
    κ_iter, κ_MAP, κ_logpost = viscomcmcsample(ntau, data_κ, mat_info_κ, prior_type[1], sigma, niter)

    println("Starting anlysis of mu with $ntau series... ")
    μ_iter, μ_MAP, μ_logpost = viscomcmcsample(ntau, data_μ, mat_info_μ, prior_type[2], sigma, niter)

    κ_result = [κ_iter, κ_MAP, κ_logpost]
    μ_result = [μ_iter, μ_MAP, μ_logpost]

    return κ_result, μ_result
end

function viscomcmc_fullanlysis(mat_label, max_ntau, data_list, results_path, prior_type = [0, 0], sigma = 0.05, niter = 1000)

    data_raw = read_datalist(data_list)
    
    result_dir, data_sring = create_result_dir(mat_label, results_path)

    open(joinpath(result_dir,"README.txt"), "w") do io
        write(io, "Host name: "*gethostname()*"\n")
        write(io, "Simulation started at: "*data_sring*"\n")
        write(io, "Materials : "*mat_label*"\n")
        write(io, "Iteration: $niter\n")
        write(io, "Sigma: $sigma\n")
        write(io, "Prior type: $prior_type\n")
    end;

    Threads.@threads for ntau in 0:max_ntau

        κ_result, μ_result = visco3Dbayes(mat_label, data_raw, ntau, prior_type, sigma, niter)
        data_out = export_dataresult(κ_result[2], μ_result[2], data_raw)

        save_csvfiles(κ_result, μ_result, ntau, data_out, result_dir)
    end

end

function viscomcmc_fullanlysis_one(mat_label, ntau_list, data_list, results_path, prior_type = [0, 0], sigma = 0.05, niter = 1000)

    data_raw = read_datalist(data_list)
    
    result_dir, data_sring = create_result_dir(mat_label, results_path)

    open(joinpath(result_dir,"README.txt"), "w") do io
        write(io, "Host name: "*gethostname()*"\n")
        write(io, "Simulation started at: "*data_sring*"\n")
        write(io, "Materials : "*mat_label*"\n")
        write(io, "Iteration: $niter\n")
        write(io, "Sigma: $sigma\n")
        write(io, "Prior type: $prior_type\n")
    end;

    Threads.@threads for ntau in ntau_list

        κ_result, μ_result = visco3Dbayes(mat_label, data_raw, ntau, prior_type, sigma, niter)
        data_out = export_dataresult(κ_result[2], μ_result[2], data_raw)

        save_csvfiles(κ_result, μ_result, ntau, data_out, result_dir)
    end

end