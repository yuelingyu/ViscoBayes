function plottheta(θ, color = "b")
    θ = exp.(θ)
    nparms = Int((length(θ) - 1)/2)
    c₀ = θ[1]
    c = θ[2:(nparms + 1)]
    ω = θ[(nparms + 2) : (2*nparms + 1)]
    PyPlot.stem(ω, c, linefmt = color, markerfmt = color * "o")
    xscale("log")
end

function getiterparam(sim, name_param::OrderedDict)
    nparam = length(name_param)
    niter = size(sim.value)[1]
    parm_iter = Array{Any}(undef, nparam)
    n = 0
    for idparam in name_param
        if idparam[2] == 1
            n += 1
            name_parm = String(idparam[1])
            parm_iter[n] =  reduce(vcat, sim[:,name_parm,:].value)
        else
            n += 1
            parmv = zeros(niter, idparam[2])
            name_parm = [String(idparam[1])*"[$i]" for i in 1:idparam[2]]
            for i = 1:idparam[2]
                parmv[:,i] = reduce(vcat, sim[:,name_parm[i],:].value)
            end
            parm_iter[n] = parmv
        end
    end
    return hcat(parm_iter...)
end


function estimateMAP(logf, θ_iter)
    niter = size(θ_iter, 1)
    ll = [logf(θ_iter[i,:]) for i = 1:niter]
    θ_MAP = θ_iter[argmax(ll),:]
    return θ_MAP, ll
end


function export_dataresult(log_κ, log_μ, data_raw)
    ndata = length(data_raw)
    result = Array{Array{Float64,2},1}(undef, ndata)
    for i in 1:ndata
        time = data_raw[i][:,1]
        σt = data_raw[i][:,2]
        ε11_expe = data_raw[i][:,3]
        ε22_expe = data_raw[i][:,4]
        εdagger1_simu = modelProny(log_μ, time, σt)
        εdagger2_simu = modelProny(log_κ, time, σt)
        ε11_simu = (1/3) * (εdagger1_simu + εdagger2_simu/3)
        ε22_simu = (1/3) * (εdagger2_simu / 3 - εdagger1_simu/2)
        result[i] = [time σt ε11_expe ε22_expe ε11_simu ε22_simu]
    end
    return result
end


function create_result_dir(testMat, results_path)
    data_label = Dates.format(now(), "Y_m_d_HH_MM_S")
    data_sring = Dates.format(now(), "Y-m-d HH:MM:S")

    result_dir = joinpath(results_path, testMat*"_"*data_label)

    if !isdir(result_dir)
        mkdir(result_dir)
    end
    return result_dir, data_sring
end



function save_csvfiles(κ_result, μ_result, ntau, data_out, result_dir)

    series_label = "$ntau"*"series"

    df_κ_iter = DataFrame(Tables.table(κ_result[1], header = Symbol.(:x, axes(κ_result[1], 2))))
    CSV.write(joinpath(result_dir, "param_iter_kappa_"*series_label*".csv"), df_κ_iter)

    kappa_MAP = vectomat(κ_result[2])
    df_kappa_MAP = DataFrame(Tables.table(kappa_MAP, header = Symbol.(:theta_MAP, axes(kappa_MAP, 2))))
    CSV.write(joinpath(result_dir, "param_MAP_kappa_"*series_label*".csv"), df_kappa_MAP)

    kappa_logpost = vectomat(κ_result[3])
    df_kappa_logpost = DataFrame(Tables.table(kappa_logpost, header = Symbol.(:logpost, axes(kappa_logpost, 2))))
    CSV.write(joinpath(result_dir, "logpost_kappa_"*series_label*".csv"), df_kappa_logpost)

    df_μ_iter = DataFrame(Tables.table(μ_result[1], header = Symbol.(:x, axes(μ_result[1], 2))))
    CSV.write(joinpath(result_dir, "param_iter_mu_"*series_label*".csv"), df_μ_iter)

    mu_MAP = vectomat(μ_result[2])
    df_mu_MAP = DataFrame(Tables.table(mu_MAP, header = Symbol.(:theta_MAP, axes(mu_MAP, 2))))
    CSV.write(joinpath(result_dir, "param_MAP_mu_"*series_label*".csv"), df_mu_MAP)

    mu_logpost = vectomat(μ_result[3])
    df_mu_logpost = DataFrame(Tables.table(mu_logpost, header = Symbol.(:logpost, axes(mu_logpost, 2))))
    CSV.write(joinpath(result_dir, "logpost_mu_"*series_label*".csv"), df_mu_logpost)


    # for i in 1:length(data_out)
    #     data_i = data_out[i]
    #     df_data = DataFrame(Tables.table(data_i, header = Symbol.(:x, axes(data_i, 2))))
    #     rename!(df_data, [:time, :stress, :eps11_expe, :eps22_expe, :eps11_simu, :eps22_simu])
    #     CSV.write(joinpath(result_dir, "results_"*series_label*"_data"*"$i"*".csv"), df_data)
    # end

end

# =============================================================================
# READ RESULT FORM SAVED FILES
# =============================================================================

function get_θ_MAP(result_folder, param_name, ntau)
    file_name = "param_MAP_"*param_name*"_"*string(ntau)*"series.csv"
    θ_file = joinpath(result_folder, file_name)
    return vec(Matrix{Float64}(DataFrame(CSV.File(θ_file))))
end

function get_θ_iter(result_folder, param_name, ntau)
    file_name = "param_iter_"*param_name*"_"*string(ntau)*"series.csv"
    θ_file = joinpath(result_folder, file_name)
    return Matrix{Float64}(DataFrame(CSV.File(θ_file)))
end

function get_logpost(result_folder, param_name, ntau)
    file_name = "logpost_"*param_name*"_"*string(ntau)*"series.csv"
    θ_file = joinpath(result_folder, file_name)
    return Matrix{Float64}(DataFrame(CSV.File(θ_file)))
end