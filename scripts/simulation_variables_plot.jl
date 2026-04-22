# Requer a instalação dos pacotes: DifferentialEquations, Plots, JLD2
using DifferentialEquations
using Plots
using JLD2
using LinearAlgebra
using Random

gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)

# Parâmetros Base
const N = 20
const tspan = (0.0, 50.0)

# Inicialização de topologia (Fixa para toda a varredura para garantir comparação justa)
Random.seed!(42)
const omega = 1.0*ones(N) #randn(N) * 0.5
const A = Float64.(rand(N, N) .< 0.3)
const B = Float64.(rand(N, N, N) .< 0.1)

# Função do sistema dinâmico
function kuramoto_dinamico!(du, x, p, t)
    N_osc = p.N
    θ = @view x[1:N_osc]
    U = reshape(@view(x[N_osc+1:end]), N_osc, N_osc)
    
    dθ = @view du[1:N_osc]
    dU = reshape(@view(du[N_osc+1:end]), N_osc, N_osc)
    
    for i in 1:N_osc
        soma_fase = 0.0
        for j in 1:N_osc
            soma_fase += U[i, j] * sin(θ[j] - θ[i])
        end
        dθ[i] = p.omega[i] + soma_fase
        
        for j in 1:N_osc
            soma_campo = 0.0
            for k in 1:N_osc
                soma_campo += p.B[i, j, k] * cos(θ[k] - θ[i])
            end
            dU[i, j] = (-U[i, j] + p.K1 * p.A[i, j] + p.K2 * soma_campo) / p.tau
        end
    end
end

# Arrays de parâmetros para a varredura
K1_vals = [0.1, 0.5, 1.0]
K2_vals = [0.0, 0.5, 1.0]
tau_vals = [0.01, 0.5, 2.0] # Inclui limite rápido e lento

for K1 in K1_vals, K2 in K2_vals, tau in tau_vals
    println("Simulando: K1=$K1 | K2=$K2 | tau=$tau")
    
    # Gerenciar diretórios específicos para esta configuração
    config_name = "K1_$(K1)_K2_$(K2)_tau_$(tau)"
    data_dir = joinpath("data", config_name)
    fig_dir = joinpath("figures", config_name)
    
    mkpath(data_dir)
    mkpath(fig_dir)

    params = (N=N, omega=omega, A=A, B=B, K1=K1, K2=K2, tau=tau)

    # Condições iniciais
    theta_0 = rand(N) .* 2π
    U_0 = K1 .* A 
    x0 = vcat(theta_0, vec(U_0))

    # Solver tolerante a stiffness para valores pequenos de tau
    prob = ODEProblem(kuramoto_dinamico!, x0, tspan, params)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=0.1)

    # Processamento de Dados
    tempos = sol.t
    theta_sol = zeros(N, length(tempos))
    U_media_in = zeros(N, length(tempos))

    for (idx, t) in enumerate(tempos)
        estado = sol.u[idx]
        theta_sol[:, idx] .= sin.(estado[1:N]) 
        
        U_matrix = reshape(estado[N+1:end], N, N)
        U_media_in[:, idx] .= dropdims(sum(U_matrix, dims=2), dims=2) ./ N
    end

    # Salvar Cache
    @save joinpath(data_dir, "simulacao_kuramoto.jld2") tempos theta_sol U_media_in params

    # Plotagem
    p1 = heatmap(tempos, 1:N, theta_sol, 
                 color=:viridis, xlabel="Tempo", ylabel="Oscilador i", 
                 title="Fases sin(θ)", clims=(-1, 1))

    p2 = heatmap(tempos, 1:N, U_media_in, 
                 color=:inferno, xlabel="Tempo", ylabel="Oscilador i", 
                 title="Força de Transmissão Média (u)")

    plt = plot(p1, p2, layout=(2, 1), size=(800, 600))
    savefig(plt, joinpath(fig_dir, "raster_plots.png"))
end

println("Varredura de parâmetros concluída.")