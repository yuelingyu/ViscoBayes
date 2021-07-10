function modelProny(log_θ, time, σt)
    θ = exp.(log_θ)
    ntau = Int((length(θ) - 1)/2)
    s0 = θ[1]
    s = θ[2 : (ntau + 1)]
    λ = θ[(ntau + 2) : (2*ntau + 1)]

    return creepRecursive(s0, s, λ, time, σt)
end


# =============================================================================
# Compute logposterior function
# =============================================================================

function log_posterior(log_θ, data, info_prior)
    ndata = length(data)
    logf = 0.0
    for i in 1:ndata
        t = data[i][:,1]
        σ = data[i][:,2]
        ε = data[i][:,3]
        logf += indiv_log_posterior(log_θ, info_prior, t, σ, ε)
    end
    return logf
end


function log_posterior(log_θ, data, info_prior, prior_type::Int64)
    ndata = length(data)
    logf = 0.0
    for i in 1:ndata
        t = data[i][:,1]
        σ = data[i][:,2]
        ε = data[i][:,3]
        logf += indiv_log_posterior(log_θ, info_prior, prior_type, t, σ, ε)
    end
    return logf
end

function indiv_log_posterior(log_θ, info_prior, prior_type::Int64, t, σ, ε)
    return log_likelihood(log_θ, t, σ, ε) + log_prior_sel(log_θ, info_prior, prior_type)
end


function log_prior_sel(log_θ, info_prior, prior_type::Int64)
    prior_type == 0 && return 0.0
    prior_type == 1 && return properties_constraints(log_θ, info_prior)
    prior_type == 2 && return tau_constraints(log_θ)
    prior_type == 3 && return properties_constraints(log_θ, info_prior) + tau_constraints(log_θ)
end


function indiv_log_posterior(log_θ, info_prior, t, σ, ε)
    return log_likelihood(log_θ, t, σ, ε) + log_prior(log_θ, info_prior)
end


function log_prior(log_θ, info_prior::Vector)

    # return properties_constraints(log_θ, info_prior) + tau_constraints(log_θ)
    # return properties_constraints(log_θ, info_prior)
    return tau_constraints(log_θ)
    # return 0.0
end


function properties_constraints(log_θ, info_prior::Vector)
    """
    In this version, we set info_prior with the form (mean_property, s_property)
    """
    c = 1.0 / exp(log_θ[1])
    pd = Normal.(info_prior[1], info_prior[2])
    return logpdf.(pd, c)
end


function tau_constraints(log_θ)
    ntau = Int((length(log_θ) - 1)/2)
    τ = log_θ[(ntau + 2) : (2*ntau + 1)]
    prod(diff(τ)) > 0 ? log_prior = -sum(τ) : log_prior = -Inf 
    # prod(diff(τ)) > 0 ? log_prior = 0.0 : log_prior = -Inf 
    return log_prior
end


function log_likelihood(log_θ, t, x, y)
    μ = modelProny(log_θ, t, x)
    # !any(isnan, μ) ? nothing : μ = zeros(size(y))
    a = length(y) / 2.0
    b = sum(abs2, y - μ) / 2.0
    b > zero(b) ? nothing : b = 1.0
    s2 = rand(InverseGamma(a, b))
    pd = Normal.(μ, sqrt(s2))
    return sum(logpdf.(pd, y))
end
