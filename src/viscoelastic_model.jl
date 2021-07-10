function materials_info(testMat::String)
    matName = Symbol(testMat)

    matDB = Dict{Symbol, Any}(
            :pp => [[1500, 2000], 0.43],
            :pmma => [[2400, 3400], 0.36]
            )

    E, ν = matDB[matName]

    κ = zeros(2)
    μ = zeros(2)

    for i in 1:length(E)
        κ[i], μ[i] = Ev_to_kappamu(E[i], ν)
    end

    mean_κ, var_κ = meanvar_conint(κ, 0.5)
    mean_μ, var_μ = meanvar_conint(μ, 0.5)

    return [mean_κ, var_κ], [mean_μ, var_μ]
end


function meanvar_conint(E::Vector, coeff = 1.0)
    μ = mean(E)
    σ = (maximum(E) - μ) / coeff
    return μ, σ
end

function vectomat(v)
    return reshape(v, length(v), 1)
end

function vectomat(v)
    return reshape(v, length(v), 1)
end

function computeCt(θ, t)
    ntau = Int((length(θ) - 1)/2)
    c₀ = θ[1]
    c = θ[2:(ntau + 1)]
    ω = θ[(ntau + 2) : (2*ntau + 1)]

    Ct = ones(size(t)).*c₀
    for i = 1:ntau
        @. Ct +=  c[i]*exp(-t*ω[i])
    end
    return Ct
end

function computeSt(θ, t)
    ntau = Int((length(θ) - 1)/2)
    s0 = θ[1]
    s = θ[2:(ntau + 1)]
    τ = θ[(ntau + 2) : (2*ntau + 1)]

    St = ones(size(t)).*s0
    for i = 1:ntau
        @. St +=  s[i]*( 1 - exp(-t*τ[i]))
    end
    return St
end

function creepRecursive(s0, s, λ, time, σt)
    ntau = length(λ)
    T = length(time)
    εt = zeros(size(σt))
    qm = zeros(T, ntau)
    Δt = diff(time)
    
    for n in 2:T
        dt = Δt[n-1]
        γ = (1 .- exp.(-λ.*dt)) ./ (λ.*dt)
        qm[n,:] = qm[n-1,:] .* exp.(-λ.*dt) .+ (σt[n] - σt[n-1]).*γ
        εt[n] = s0*σt[n] .+ sum(s.*( σt[n] .- σt[1] .-  qm[n,:]))
    end

    return εt
end


function matricesrelax(C0, C, ω)
    N = length(ω)
    D = size(C0,1)
    B = Matrix{Float64}(I, D*N, D*N)
    L1 = sum(C) + C0
    L2 = hcat([cholesky(ω[i]*C[i]).L for i in 1:N]...)
    L3 = Diagonal(vcat([ω[i]*ones(D) for i in 1:N]...))
    return B, L1, L2, L3
end

function matricescreep(S0, S, λ)
    N = length(λ)
    D = size(S0, 1)
    # B = Matrix{Float64}(I, D*N, D*N)
    B = Diagonal(ones(D*N))
    A1 = S0
    A2 = hcat([cholesky(λ[i]*S[i]).L for i in 1:N]...)
    A3 = Diagonal(vcat([λ[i]*ones(D) for i in 1:N]...))
    return B, A1, A2, A3
end



############################################################################################
# Relaxation
############################################################################################


function relaxEulerExp(C0, C, ω, ε, t)
    B, L1, L2, L3 = matricesrelax(C0, C, ω)
    D = size(C0,1)
    σ = zeros(D, length(t))
    ξ = zeros(D*length(ω), length(t))
    B1 = inv(B)
    εtmp = ε[:,1]
    σtmp = L1*εtmp
    σ[:,1] = σtmp

    for i = 2:length(t)
        h = t[i] - t[i-1]
        W1 = I - h*B1*L3
        W2 = -h*B1*transpose(L2)
        ξtmp = W1*ξ[:,i-1] + W2*ε[:,i-1]
        σ[:,i] = L1*ε[:,i-1] + L2*ξtmp
        ξ[:,i] = ξtmp
    end
    return σ
end

function relaxEulerImp(C0, C, ω, ε, t)
    B, L1, L2, L3 = matricesrelax(C0, C, ω)
    D = size(C0,1)
    σ = zeros(D, length(t))
    ξ = zeros(D*length(ω), length(t))
    B1 = inv(B)
    εtmp = ε[:,1]
    σtmp = L1*εtmp
    σ[:,1] = σtmp

    for i = 2:length(t)
        h = t[i] - t[i-1]
        W1 = inv(I + h*B1*L3)
        W2 = (I + h*B1*L3)\(-h*B1*transpose(L2))
        ξtmp = W1*ξ[:,i-1] + W2*ε[:,i]
        σ[:,i] = L1*ε[:,i] + L2*ξtmp
        ξ[:,i] = ξtmp
    end
    return σ
end

function relaxCrankNicolson(C0, C, ω, ε, t)
    B, L1, L2, L3 = matricesrelax(C0, C, ω)
    D = size(C0,1)
    σ = zeros(D, length(t))
    ξ = zeros(D*length(ω), length(t))
    B1 = inv(B)
    εtmp = ε[:,1]
    σtmp = L1*εtmp
    σ[:,1] = σtmp

    for i = 2:length(t)
        h = t[i] - t[i-1]
        W1 = (I + (h/2)*B1*L3)\(I - (h/2)*B1*L3)
        W2 = (I + (h/2)*B1*L3)\(-(h/2)*B1*transpose(L2))
        ξtmp = W1*ξ[:,i-1] + W2*(ε[:,i] + ε[:,i-1])
        σ[:,i] = L1*ε[:,i] + L2*ξtmp
        ξ[:,i] = ξtmp
    end
    return σ
end


############################################################################################
# Creep
############################################################################################


function creepEulerExp(S0, S, λ, σ, t)
    B, A1, A2, A3 =  matricescreep(S0, S, λ)
    D = size(S0,1)
    ε = zeros(D, length(t))
    χ = zeros(D*length(λ), length(t))
    B1 = inv(B)
    σtmp = σ[:,1]
    εtmp = A1*σtmp
    ε[:,1] = εtmp

    for i = 2:length(t)
        h = t[i] - t[i-1]
        W1 = I - h*B1*A3
        W2 = -h*B1*transpose(A2)
        χtmp = W1*χ[:,i-1] + W2*σ[:,i-1]
        ε[:,i] = A1*σ[:,i-1] - A2*χtmp
        χ[:,i] = χtmp
    end
    return ε
end

function creepEulerImp(S0, S, λ, σ, t)
    B, A1, A2, A3 =  matricescreep(S0, S, λ)
    D = size(S0,1)
    ε = zeros(D, length(t))
    χ = zeros(D*length(λ), length(t))
    B1 = inv(B)
    σtmp = σ[:,1]
    εtmp = A1*σtmp
    ε[:,1] = εtmp

    for i = 2:length(t)
        h = t[i] - t[i-1]
        W1 = inv(I + h*B1*A3)
        W2 = (I + h*B1*A3)\(-h*B1*transpose(A2))
        χtmp = W1*χ[:,i-1] + W2*σ[:,i]
        ε[:,i] = A1*σ[:,i] - A2*χtmp
        χ[:,i] = χtmp
    end
    return ε
end

function creepCrankNicolson(S0, S, λ, σ, t)
    B, A1, A2, A3 =  matricescreep(S0, S, λ)
    D = size(S0,1)
    ε = zeros(D, length(t))
    χ = zeros(D*length(λ), length(t))
    B1 = inv(B)
    σtmp = σ[:,1]
    εtmp = A1*σtmp
    ε[:,1] = εtmp

    for i = 2:length(t)
        h = t[i] - t[i-1]
        W1 = (I + (h/2)*B1*A3)\(I - (h/2)*B1*A3)
        W2 = (I + (h/2)*B1*A3)\(-(h/2)*B1*transpose(A2))
        χtmp = W1*χ[:,i-1] + W2*(σ[:,i] + σ[:,i-1])
        ε[:,i] = A1*σ[:,i] - A2*χtmp
        χ[:,i] = χtmp
    end
    return ε
end


# =============================================================================
# Interconvention S(t) <-> C(t)
# =============================================================================

function interconv_ctos(s0, s, τ)
    """
    Interconvention s(t) -> c(t) 1D
    """
    ntau = length(τ)
    _, A1, A2, A3 = matricescreep(s0, s, τ)

    L1 = inv(A1)
    L2T = A1\A2'
    L3 = A3 + transpose(A2)/A1*A2

    P, _, _ = svd(L3)

    L2star = vec(transpose(P)*L2T)
    L3star = transpose(P)*L3*P

    ctau = diag(L3star)
    c = L2star.*L2star./ctau
    c_sum = sum(c)

    c0 = L1 - c_sum

    return c0, c, ctau
end


function kappamu_to_Ev(κ, μ)
    E = 9*κ*μ / (3*κ + μ)
    ν = (3*κ - 2*μ) / (2 * (3*κ + μ))
    return E, ν
end

function Ev_to_kappamu(E, ν)
    κ = E / (3*(1 - 2*ν))
    μ = E / (2*(1 + ν))
    return κ, μ
end