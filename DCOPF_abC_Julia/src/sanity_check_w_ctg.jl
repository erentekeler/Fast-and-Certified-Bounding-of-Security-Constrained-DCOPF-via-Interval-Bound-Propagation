using Pkg # I always need to specify the env that I want to use when I am calling it from the console
Pkg.activate("DCOPF_abC_Julia")

using Revise
using QuasiGrad
using SparseArrays
using LinearAlgebra
using XLSX
using DataFrames
using JuMP
using Gurobi
using JSON

# identify the data
function runSanityCheck(network_file_name, output_file_name, dtype)
    # call the jsn data
    jsn = QuasiGrad.load_json("DCOPF_abC_Julia/data/$(network_file_name)")

    # initialize the network 
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, Div=1, hpc_params=false);

    function kys(param)
        tuple = fieldnames(typeof(param))
        for el in tuple
            println(el)
        end
    end

    ## Those need to be run once only
    # what matrices do we need?
    ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]
    Ybs         = QuasiGrad.spdiagm(ac_b_params)
    Yflow       = Ybs*ntk.E # flow matrix
    Yb          = ntk.Yb    # Ybus matrix
    Ybr         = Yb[2:end,2:end]  # use @view ? 
    E           = ntk.E
    Er          = E[:,2:end]
    Yfr         = Ybs*Er
    ptdf        = Ybs*Er*inv(Matrix(Er'*Ybs*Er))
    ptdf        = [zeros(sys.nl + sys.nx) ptdf]

    # flow limits
    pf_max_base = 0.083*[prm.acline.mva_ub_nom; prm.xfm.mva_ub_nom]
    pf_max_ctg  = 0.083*[prm.acline.mva_ub_em;  prm.xfm.mva_ub_em]

    N_d2inj = zeros(sys.nb,sys.ndev)
    # loop over consumers (loads)
    for bus in 1:sys.nb
        for cs in idx.cs[bus]
            N_d2inj[bus,cs] = -1
        end

        # loop over producers (generators)
        for pr in idx.pr[bus]
            N_d2inj[bus,pr] = +1
        end
    end

    # prepare contingency vectors
    u_k = [zeros(sys.nb-1) for ctg_ii in 1:sys.nctg]
    g_k = zeros(sys.nctg)
    z_k = [zeros(sys.nac) for ctg_ii in 1:sys.nctg]

    for ctg_ii in 1:sys.nctg
        ln_ind          = ntk.ctg_out_ind[ctg_ii][1]
        ac_b_params_ctg = -[prm.acline.b_sr; prm.xfm.b_sr]
        ac_b_params_ctg[ln_ind] = 0

        Ybs_ctg       = QuasiGrad.spdiagm(ac_b_params_ctg)
        Yfr_ctg       = Ybs_ctg*Er # flow matrix

        # apply sparse 
        ei = Array(Er[ln_ind,:])
        
        # compute u, g, and z!
        u_k[ctg_ii]  = Ybr\ei
        g_k[ctg_ii]  = -ac_b_params[ln_ind]/(1.0+(dot(ei,u_k[ctg_ii]))*-ac_b_params[ln_ind])
        mul!(z_k[ctg_ii], Yfr_ctg, u_k[ctg_ii])
    end
    ## Rest needs to be run for each time index


    # %% set up a SCOPF problem -- assume all generators are on :)
    sol_and_timing = zeros(size(prm.ts.time_keys,1), 3) # Computation times and solutions for each time index
    dev_power_sol = zeros(size(prm.ts.time_keys,1), sys.ndev) # Solutions for each time index, rows are time indices columns are the device dispatches
    # time index
    for tii in prm.ts.time_keys
        println("Running time index $(tii) for $(network_file_name).")
        dt   = prm.ts.duration[tii] # duration
        p_lb = zeros(sys.ndev)
        p_ub = zeros(sys.ndev)

        # upper and lower bounds for loads/gens
        for dev in prm.dev.dev_keys
            p_lb[dev] = prm.dev.p_lb[dev][tii]
            p_ub[dev] = prm.dev.p_ub[dev][tii]
        end


        #here the network parameters and information are stored in a df and then written into the xlsx file
        if tii == 1
            info_df = DataFrame(Dict(
                "# time indices" => [size(prm.ts.time_keys, 1)],
                "# contingencies" => [sys.nctg],
                "# buses" => [sys.nb],
                "# lines" => [size(ac_b_params, 1)],
                "# devices" => [sys.ndev],
                "# cost sectors" => [maximum(unique([length([prm.dev.cum_cost_blocks[dix][tii][1] for dix in 1:sys.ndev][dix]) for dix in 1:sys.ndev]))],
                # Some cases have devices having different amount of sectors than the other, if that is the case, I want to know
                "# cost parameter mismatch" => length(unique([length([prm.dev.cum_cost_blocks[dix][tii][1] for dix in 1:sys.ndev][dix]) for dix in 1:sys.ndev]))!=1 ? "yes" : "no",
                "dtype" => dtype
            ))

            XLSX.openxlsx("DCOPF_abC_Python/output/$(output_file_name)", mode="w") do file
                sheet = XLSX.addsheet!(file, "Net info")
                sheet1 = XLSX.addsheet!(file, "Gurobi")
                sheet2 = XLSX.addsheet!(file, "dev_powers")
                XLSX.writetable!(sheet, eachcol(info_df), names(info_df))
            end
        end


        # %% next!
        """ Model 4: now, add contingencies (i.e., lines losses)
        """
        start_time = time()
        model = Model(Gurobi.Optimizer)

        # device power bounds!
        @variable(model,            dev_power[1:sys.ndev])
        @constraint(model, p_lb .<= dev_power .<= p_ub)

        # get injections
        pinj = N_d2inj * dev_power

        # for each device, assign a set of power blocks, each one bounded
        dev_power_blocks = [@variable(model, [blk = 1:length(prm.dev.cost[dev][tii])], lower_bound = 0, upper_bound = prm.dev.cum_cost_blocks[dev][tii][2][blk+1]) for dev in 1:sys.ndev]

        dev_cost = Vector{AffExpr}(undef, sys.ndev)
        for dev in 1:sys.ndev
            dev_cost[dev] = AffExpr(0.0)
            # the device power is the sum of the blocks
            @constraint(model, dev_power[dev] == sum(dev_power_blocks[dev]))

            # now, get the cost!
            cst = prm.dev.cum_cost_blocks[dev][tii][1][2:end]
            if prm.dev.device_type[dev] == "producer"
                # this is a generator
                dev_cost[dev] = -dt*sum(dev_power_blocks[dev].*cst)
            elseif prm.dev.device_type[dev] == "consumer"
                # this is a load
                dev_cost[dev] = dt*sum(dev_power_blocks[dev].*cst)
            else
                println("device not found!")
            end
        end

        # power flow!
        @variable(model, t[1:(sys.nl + sys.nx)], lower_bound = 0.0)
        @constraint(model,  ptdf*pinj - pf_max_base  .<= t)
        @constraint(model, -ptdf*pinj - pf_max_base  .<= t)

        # also penalize power imbalance
        @variable(model, tb[1:sys.nb], lower_bound = 0.0)
        @constraint(model,   E'*ptdf*pinj - pinj .<= tb)
        @constraint(model,   pinj - E'*ptdf*pinj .<= tb)


        # now, rank-1 correct to get contingency flows
        M_mask = Matrix(I, sys.nl+sys.nx, sys.nl+sys.nx)
        tf_ctg = [@variable(model, [1:(sys.nl+sys.nx)], lower_bound = 0.0) for ii in 1:sys.nctg]
        z_ctg  = AffExpr(0.0)

        for ii in 1:sys.nctg #sys.nctg
            # rank 1 correct:
            if ii > 1
                past_ln_ind                     = ntk.ctg_out_ind[ii-1][1]
                M_mask[past_ln_ind,past_ln_ind] = 1.0
            end
            ln_ind                = ntk.ctg_out_ind[ii][1]
            M_mask[ln_ind,ln_ind] = 0.0
            pflow_ctg = (M_mask*ptdf)*pinj .- z_k[ii].*(g_k[ii]*dot(u_k[ii], pinj[2:end]))

            # now, get flow penalties
            @constraint(model,    pflow_ctg - pf_max_ctg  .<= tf_ctg[ii])
            @constraint(model,   -pflow_ctg - pf_max_ctg  .<= tf_ctg[ii])
            add_to_expression!(z_ctg, sum(tf_ctg[ii]))

        end

        market_surplus = sum(dev_cost) - dt*prm.vio.s_flow*sum(t) - dt*prm.vio.p_bus*sum(tb) - dt*prm.vio.s_flow*z_ctg
        @objective(model, Max, market_surplus)

        # optimize
        optimize!(model)

        println("========")
        println(objective_value(model))
        println("========")

        obj_val_model = objective_value(model)

        # your code here
        elapsed = time() - start_time

        sol_and_timing[tii, 1] = tii
        sol_and_timing[tii, 2] = elapsed
        sol_and_timing[tii, 3] = obj_val_model  
        
        dev_power_sol[tii, :] = value.(dev_power) # Each time index solution is saved for forward pass check
        
        # computation times and the objective values are saved here
        colnames = ["Time Index", "Computation time", "Objective value"]
        sol_and_timing_df = DataFrame(sol_and_timing, colnames)

        # Dispatch for each time index is saved in the dev_powers sheet
        sol_df = DataFrame(dev_power_sol, "dev_" .* string.(1:sys.ndev))

        XLSX.openxlsx("DCOPF_abC_Python/output/$(output_file_name)", mode="rw") do file
            sheet = file["Gurobi"]
            XLSX.writetable!(sheet, eachcol(sol_and_timing_df), names(sol_and_timing_df))
        end

        XLSX.openxlsx("DCOPF_abC_Python/output/$(output_file_name)", mode="rw") do file
            sheet = file["dev_powers"]
            XLSX.writetable!(sheet, eachcol(sol_df), names(sol_df))
        end
    
    end
        
end

# Main
# Read the json file that is passed as an argument in the subprocess
args = JSON.parse(ARGS[1])
cases = args["cases"]
dtype = args["dtype"]
print(cases)
for case in cases
    output_file_name = replace(case, ".json" => ".xlsx")
    runSanityCheck(case, output_file_name, dtype)
end