# scripts/plot_training.jl
#
# Plot a training trajectory CSV produced by examples/case57_train.jl.
#
# Usage:
#   julia --project=examples scripts/plot_training.jl outputs/training_log_case57_<ts>.csv
#
# Produces a PNG next to the CSV.

using DelimitedFiles
using Plots
gr()

function main(csv_path::String)
    @assert isfile(csv_path) "CSV not found: $csv_path"
    data, header = readdlm(csv_path, ',', Float64; header = true)
    cols = Dict(strip(c) => i for (i, c) in enumerate(header))

    iters = data[:, cols["outer_iter"]]
    ρ        = data[:, cols["rho"]]
    mean_μ   = data[:, cols["mean_abs_mu"]]
    mean_λ   = data[:, cols["mean_abs_lambda"]]
    mean_v   = data[:, cols["mean_violations"]]
    max_vi   = data[:, cols["max_violation_ineq"]]
    max_ve   = data[:, cols["max_violation_eq"]]
    max_b    = data[:, cols["max_bound_violation"]]
    mean_o   = data[:, cols["mean_objs"]]
    primal_l = data[:, cols["primal_loss"]]

    p1 = plot(iters, primal_l, xlabel="outer iter", ylabel="primal AL loss", title="Primal loss", lw=2)
    p2 = plot(iters, ρ,        xlabel="outer iter", ylabel="ρ",        title="Penalty ρ", lw=2)
    p3 = plot(iters, [mean_μ mean_λ], xlabel="outer iter",
        label=["mean |μ|" "mean |λ|"], title="Dual magnitudes", lw=2)
    p4 = plot(iters, [max_vi max_ve max_b], xlabel="outer iter",
        label=["max V_ineq" "max V_eq" "max V_bound"],
        title="Max violations (log)", yscale=:log10, lw=2)
    p5 = plot(iters, mean_v,   xlabel="outer iter", ylabel="mean violations", title="Mean violations", lw=2)
    p6 = plot(iters, mean_o,   xlabel="outer iter", ylabel="mean objective", title="Mean cost", lw=2)

    fig = plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1400, 800), legend=:topright)
    png_path = replace(csv_path, r"\.csv$" => ".png")
    savefig(fig, png_path)
    @info "Saved" png_path
end

if length(ARGS) < 1
    error("usage: julia --project=examples scripts/plot_training.jl <csv_path>")
end
main(ARGS[1])
