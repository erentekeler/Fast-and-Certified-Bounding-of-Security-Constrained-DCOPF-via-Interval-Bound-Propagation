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
function exportNetworkInfo(network_file_name, output_file_name)
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


    # time index
    tii = 1
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
        ))

        XLSX.openxlsx("DCOPF_abC_Python/output/$(output_file_name)", mode="w") do file
            sheet = XLSX.addsheet!(file, "Net info")
            XLSX.writetable!(sheet, eachcol(info_df), names(info_df))
        end
    end

end


# identify the data
# I am passing the file path and the created file names as an argument, looping over them here
network_file_name = ARGS[1]
output_file_name = ARGS[2]
exportNetworkInfo(network_file_name, output_file_name)