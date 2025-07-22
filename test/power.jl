function _get_case_file(filename::String)
    isfile(filename) && return filename

    cached = joinpath(PGLib.PGLib_opf, filename)
    !isfile(cached) && error("File $filename not found in PGLib/pglib-opf")
    return cached
end
function _build_power_ref(filename)
    path = _get_case_file(filename)
    data = PowerModels.parse_file(path)
    PowerModels.standardize_cost_terms!(data, order = 2)
    PowerModels.calc_thermal_limits!(data)
    return PowerModels.build_ref(data)[:it][:pm][:nw][0]
end


_convert_array(data::N, backend) where {names,N<:NamedTuple{names}} =
    NamedTuple{names}(ExaModels.convert_array(d, backend) for d in data)
function _parse_ac_data_raw(filename; T=Float64)
    ref = _build_power_ref(filename) # FIXME: only parse once
    arcdict    = Dict(a => k for (k, a) in enumerate(ref[:arcs]))
    busdict    = Dict(k => i for (i, (k, _)) in enumerate(ref[:bus]))
    gendict    = Dict(k => i for (i, (k, _)) in enumerate(ref[:gen]))
    branchdict = Dict(k => i for (i, (k, _)) in enumerate(ref[:branch]))
    return (
        bus = [
            begin
                loads   = [ref[:load][l]   for l in ref[:bus_loads][k]]
                shunts  = [ref[:shunt][s]  for s in ref[:bus_shunts][k]]
                pd  = T(sum(load["pd"]  for load  in loads;  init = 0.0))
                qd  = T(sum(load["qd"]  for load  in loads;  init = 0.0))
                gs  = T(sum(shunt["gs"] for shunt in shunts; init = 0.0))
                bs  = T(sum(shunt["bs"] for shunt in shunts; init = 0.0))
                (i = busdict[k], pd = pd, gs = gs, qd = qd, bs = bs)
            end for (k, _) in ref[:bus]
        ],
        gen = [
            (
                i     = gendict[k],
                cost1 = T(v["cost"][1]),
                cost2 = T(v["cost"][2]),
                cost3 = T(v["cost"][3]),
                bus   = busdict[v["gen_bus"]],
            ) for (k, v) in ref[:gen]
        ],
        arc = [
            (i = k, rate_a = T(ref[:branch][l]["rate_a"]), bus = busdict[i]) for
            (k, (l, i, _)) in enumerate(ref[:arcs])
        ],
        branch = [
            begin
                branch = branch_raw
                f_idx = arcdict[i, branch["f_bus"], branch["t_bus"]]
                t_idx = arcdict[i, branch["t_bus"], branch["f_bus"]]

                g,  b  = PowerModels.calc_branch_y(branch)
                tr, ti = PowerModels.calc_branch_t(branch)
                ttm    = tr^2 + ti^2

                g_fr = branch["g_fr"]; b_fr = branch["b_fr"]
                g_to = branch["g_to"]; b_to = branch["b_to"]

                (
                    i      = branchdict[i],
                    j      = 1,
                    f_idx  = f_idx,
                    t_idx  = t_idx,
                    f_bus  = busdict[branch["f_bus"]],
                    t_bus  = busdict[branch["t_bus"]],
                    c1     = T((-g * tr - b * ti) / ttm),
                    c2     = T((-b * tr + g * ti) / ttm),
                    c3     = T((-g * tr + b * ti) / ttm),
                    c4     = T((-b * tr - g * ti) / ttm),
                    c5     = T((g + g_fr) / ttm),
                    c6     = T((b + b_fr) / ttm),
                    c7     = T((g + g_to)),
                    c8     = T((b + b_to)),
                    rate_a_sq = T(branch["rate_a"]^2),
                )
            end for (i, branch_raw) in ref[:branch]
        ],
        ref_buses = [busdict[i] for (i, _) in ref[:ref_buses]],
        vmax      = [T(v["vmax"]) for (_, v) in ref[:bus]],
        vmin      = [T(v["vmin"]) for (_, v) in ref[:bus]],
        pmax      = [T(v["pmax"]) for (_, v) in ref[:gen]],
        pmin      = [T(v["pmin"]) for (_, v) in ref[:gen]],
        qmax      = [T(v["qmax"]) for (_, v) in ref[:gen]],
        qmin      = [T(v["qmin"]) for (_, v) in ref[:gen]],
        rate_a    = [T(ref[:branch][l]["rate_a"]) for (l, _, _) in ref[:arcs]],
        angmax    = [T(b["angmax"]) for (_, b) in ref[:branch]],
        angmin    = [T(b["angmin"]) for (_, b) in ref[:branch]],
    )
end

_parse_ac_data(filename) = _parse_ac_data_raw(filename)
function _parse_ac_data(filename, backend; T=Float64)
    _convert_array(_parse_ac_data_raw(filename, T=T), backend)
end

# Parametric version

function create_parametric_ac_power_model(filename::String = "pglib_opf_case14_ieee.m";
    prod::Bool = true, backend = OpenCLBackend(), T=Float64, kwargs...)
    data = _parse_ac_data(filename, backend, T=T)
    c = ExaCore(T; backend = backend)

    va = variable(c, length(data.bus);)
    vm = variable(
            c,
            length(data.bus);
            start = fill!(similar(data.bus, T), 1.0),
            lvar = data.vmin,
            uvar = data.vmax,
        )

    pg = variable(c, length(data.gen); lvar = data.pmin, uvar = data.pmax)
    qg = variable(c, length(data.gen); lvar = data.qmin, uvar = data.qmax)

    @allowscalar load_multiplier = parameter(c, [1.0 for b in data.bus])

    p = variable(c, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)
    q = variable(c, length(data.arc); lvar = -data.rate_a, uvar = data.rate_a)

    o = objective(c, g.cost1 * pg[g.i]^2 + g.cost2 * pg[g.i] + g.cost3 for g in data.gen)

    # Reference bus angle ------------------------------------------------------
    c1 = constraint(c, va[i] for i in data.ref_buses)

    # Branch power-flow equations ---------------------------------------------
    constraint(
        c,
        (p[b.f_idx] - b.c5 * vm[b.f_bus]^2 -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch),
    )

    constraint(
        c,
        (q[b.f_idx] + b.c6 * vm[b.f_bus]^2 +
        b.c4 * (vm[b.f_bus] * vm[b.t_bus] * cos(va[b.f_bus] - va[b.t_bus])) -
        b.c3 * (vm[b.f_bus] * vm[b.t_bus] * sin(va[b.f_bus] - va[b.t_bus])) for
        b in data.branch),
    )

    constraint(
        c,
        (p[b.t_idx] - b.c7 * vm[b.t_bus]^2 -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch),
    )

    constraint(
        c,
        (q[b.t_idx] + b.c8 * vm[b.t_bus]^2 +
        b.c2 * (vm[b.t_bus] * vm[b.f_bus] * cos(va[b.t_bus] - va[b.f_bus])) -
        b.c1 * (vm[b.t_bus] * vm[b.f_bus] * sin(va[b.t_bus] - va[b.f_bus])) for
        b in data.branch),
    )

    # Angle difference limits --------------------------------------------------
    constraint(
        c,
        va[b.f_bus] - va[b.t_bus] for b in data.branch; lcon = data.angmin, ucon = data.angmax,
    )

    # Apparent power thermal limits -------------------------------------------
    constraint(
        c,
        p[b.f_idx]^2 + q[b.f_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )
    constraint(
        c,
        p[b.t_idx]^2 + q[b.t_idx]^2 - b.rate_a_sq for b in data.branch;
        lcon = fill!(similar(data.branch, Float64, length(data.branch)), -Inf),
    )

    # Power balance at each bus -----------------------------------------------
    load_balance_p = constraint(c, b.pd * load_multiplier[b.i] + b.gs * vm[b.i]^2 for b in data.bus)
    load_balance_q = constraint(c, b.qd * load_multiplier[b.i] - b.bs * vm[b.i]^2 for b in data.bus)

    # Map arc & generator variables into the bus balance equations
    constraint!(c, load_balance_p, a.bus => p[a.i]   for a in data.arc)
    constraint!(c, load_balance_q, a.bus => q[a.i]   for a in data.arc)
    constraint!(c, load_balance_p, g.bus => -pg[g.i] for g in data.gen)
    constraint!(c, load_balance_q, g.bus => -qg[g.i] for g in data.gen)

    return ExaModel(c; prod = prod), length(data.bus), length(data.gen), length(data.arc)
end
