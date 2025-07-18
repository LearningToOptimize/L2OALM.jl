module L2OALM

using BatchNLPKernels
using ExaModels

using Lux
using LuxCUDA
using Lux.Training
using MLUtils
using Optimisers
using CUDA

# using OpenCL, pocl_jll, AcceleratedKernels

function LagrangianDualLoss(;max_dual=1e6)
    return (dual_model, ps_dual, st_dual, data) -> begin
        x, dual_hat_k, gh, ρ = data
        
        # Get current dual predictions
        dual_hat, st_dual_new = dual_model(x, ps_dual, st_dual)
        
        # Separate bound and equality constraints
        gh_bound = gh[1:end-n_bus*2,:]
        gh_equal = gh[end-n_bus*2+1:end,:]
        dual_hat_bound = dual_hat_k[1:end-n_bus*2,:]
        dual_hat_equal = dual_hat_k[end-n_bus*2+1:end,:]
        
        # Target for dual variables
        dual_target = vcat(
            min.(max.(dual_hat_bound + ρ .* gh_bound, 0), max_dual),
            min.(dual_hat_equal + ρ .* gh_equal, max_dual)
        )
        
        loss = mean((dual_hat .- dual_target).^2)
        return loss, st_dual_new, (dual_loss=loss,)
    end
end

function LagrangianPrimalLoss(bm::BatchModel)    
    return (model, ps, st, data) -> begin
        Θ, dual_hat, ρ = data
        num_s = size(Θ, 2)

        # Forward pass for prediction
        X̂, st_new = model(Θ, ps, st)
        
        # Calculate violations and objectives
        objs = BNK.objective!(bm, X̂, Θ)
        # gh = constraints!(bm, X̂, Θ)
        Vc, Vb = BNK.all_violations!(bm, X̂, Θ)
        V = vcat(Vb, Vc)
        total_loss = (
            sum(abs.(dual_hat .* V)) / num_s +
            ρ / 2 * sum((V).^2) / num_s +
            mean(objs)
        )

        return total_loss, st_new, (
            total_loss=total_loss,
        )
    end
end


end
