
function t_multiplier(n, m, α)
    return quantile(TDist(n-m),1-α/2)
end

function coeff_SE(xt, yt, y_hat)
    n = length(xt)
    MSE = sqL2dist(y_hat, yt) / (n-2)
    ssq_dist = sum(abs2, xt .- mean(xt))
    return n, MSE, ssq_dist
end

function standerror_CI(xt, yt, y_hat)
    n, MSE, ssq_dist = coeff_SE(xt, yt, y_hat)
    return [sqrt(MSE * (1/n + (xi - mean(xt))^2 / ssq_dist)) for xi in xt]
end

function standerror_PI(xt, yt, y_hat)
    n, MSE, ssq_dist = coeff_SE(xt, yt, y_hat)
    return [sqrt(MSE * (1/n + 1.0 + (xi - mean(xt))^2 / ssq_dist)) for xi in xt]
end

function confidence_interval(xt, yt, y_hat, m, α)
    n = length(xt)
    return t_multiplier(n, m, α)*standerror_CI(xt, yt, y_hat)
end

function prediction_interval(xt, yt, y_hat, m, α)
    n = length(xt)
    return t_multiplier(n, m, α)*standerror_PI(xt, yt, y_hat)
end

function envelope_CI(xt, yt, y_hat, m, α)
    δ = confidence_interval(xt, yt, y_hat, m, α)
    return y_hat .-δ, y_hat .+ δ
end

function envelope_PI(xt, yt, y_hat, m, α)
    δ = prediction_interval(xt, yt, y_hat, m, α)
    return y_hat .-δ, y_hat .+ δ
end


function data_interval(result_file, data_file, ntau_κ, ntau_μ, α = 0.05)
    
    data = Matrix{Float64}(Matrix{Float64}(DataFrame(CSV.File(data_file))))

    log_κ = get_θ_MAP(result_file, "kappa", ntau_κ)
    log_μ = get_θ_MAP(result_file, "mu", ntau_μ)

    ε_dager1_simu =  modelProny(log_μ, data[:,1], data[:,2])
    ε_dager2_simu =  modelProny(log_κ, data[:,1], data[:,2])

    ε11_simu, ε22_simu =  strain_dagger_to_axial_trans(ε_dager1_simu, ε_dager2_simu)

    m = 2*(ntau_κ + ntau_μ + 1)
    δ11_CI = confidence_interval(data[:,1], data[:,3], ε11_simu, m, α)
    δ11_PI = prediction_interval(data[:,1], data[:,3], ε11_simu, m, α)

    δ22_CI = confidence_interval(data[:,1], data[:,4], ε22_simu, m, α)
    δ22_PI = prediction_interval(data[:,1], data[:,4], ε22_simu, m, α)

    return [data[:,1] data[:,3] ε11_simu δ11_CI δ11_PI data[:,4] ε22_simu δ22_CI δ22_PI]

end


function save_interval_file(interval_file, result_file, data_file, ntau_κ, ntau_μ, α = 0.05)
    data_out = data_interval(result_file, data_file, ntau_κ, ntau_μ, α)
    df = DataFrame(Tables.table(data_out, header = [:time, 
                            :eps11_expe, :eps11_simu, :delta_CI_11, :delta_PI_11,
                            :eps22_expe, :eps22_simu, :delta_CI_22, :delta_PI_22]))
    CSV.write(interval_file, df)
    return
end
