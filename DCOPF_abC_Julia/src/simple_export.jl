using Pkg # I always need to specify the env that I want to use when I am calling it from the console
Pkg.activate("DCOPF_abC_Julia")

using QuasiGrad
using LinearAlgebra
using DataFrames, HDF5, CSV



function exportNetworkFiles(network_file_name, static_params_name, ts_params_name)
    # call the jsn data
    jsn = QuasiGrad.load_json("DCOPF_abC_Julia/data/$(network_file_name)")

    # initialize the network 
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = QuasiGrad.base_initialization(jsn, Div=1, hpc_params=false);


    # what matrices do we need?
    # tii = 1 # dummy just to show
    ac_b_params = -[prm.acline.b_sr; prm.xfm.b_sr]
    Ybs         = QuasiGrad.spdiagm(ac_b_params)
    Yflow       = Ybs*ntk.E # flow matrix
    Yb          = ntk.Yb    # Ybus matrix
    Ybr         = Yb[2:end,2:end]  # use @view ? 
    E           = ntk.E
    Er          = ntk.E[:,2:end]
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


    u_k = reduce(hcat, u_k) 
    u_k = vcat(zeros(1, size(u_k,2)), u_k) # Zero padding is required for it to match the shape of Pinj

    z_k = reduce(hcat, z_k) 
    b = z_k .* g_k'

    M = ones(sys.nctg, sys.nl+sys.nx)
    for ctg_ii in 1:sys.nctg
        ln_ind = ntk.ctg_out_ind[ctg_ii][1]
        M[ctg_ii, ln_ind] = 0.0
    end



    # %% Exporting the data, Let's first export the network parameters, they are static
    static_params_file = "DCOPF_abC_Python/parameters/$(static_params_name)"

    h5open(static_params_file, "w") do file 
        file["Yflow"] = Matrix(Yflow)
        file["Yb"] = Matrix(Yb)
        file["ptdf"] = Matrix(ptdf)
        file["pf_max_base"] = pf_max_base
        file["N_d2inj"] = Matrix(N_d2inj)
        file["E"] = Matrix(E)
        file["s_flow_violation"] = prm.vio.s_flow 
        file["p_bus_violation"] = prm.vio.p_bus
        file["device_type"] = prm.dev.device_type

        # Contingency parameters are exported here
        file["M"] = M
        file["u"] = u_k
        file["b"] = b
        file["pf_max_ctg"] = pf_max_ctg
        
    end

    # I checked if they are saved correctly
    c = h5open(static_params_file, "r") do file
        read(file)
    end



    # %% Exporting time dependent parameters here TODO Ask if we wanna lose small precision here, initially used Float32 but some precision is lost
    ts_params_file = "DCOPF_abC_Python/parameters/$(ts_params_name)"

    duration_df = DataFrame(time_index = Int.(prm.ts.time_keys), duration = Float64.(prm.ts.duration)) # This dataframe keeps the durations for each time index
    P_ulb_df = DataFrame(time_index = Int[], dev = Int[], p_lb = Float64[], p_ub = Float64[]) # This dataframe keeps the upper and lower bound on the loads and generations
    sector_cst_df = DataFrame(time_index = Int[], dev = Int[], sector_no = Int[], sector_cst = Float64[]) # This dataframe keeps the sector costs of the devices
    sector_p_df = DataFrame(time_index = Int[], dev = Int[], sector_no = Int[], sector_p = Float64[]) # This dataframe keeps the sector powers of the devices


    for tii in prm.ts.time_keys # going over all time indices
        for dev in prm.dev.dev_keys # going over all devices

        # Pushing new rows to the P_ulb_df dataFrame
        push!(P_ulb_df, [tii, dev, prm.dev.p_lb[dev][tii], prm.dev.p_ub[dev][tii]])

        # Pushing new rows to the sector_cst_df dataFrame
        sector_no = 1
        for sector_cst in prm.dev.cum_cost_blocks[dev][tii][1] # Going over all sectors costs
            push!(sector_cst_df, [tii, dev, sector_no, sector_cst])
            sector_no += 1
        end

        # Pushing new rows to the sector_cst_df dataFrame
        sector_no = 1
        for sector_p in prm.dev.cum_cost_blocks[dev][tii][2] # going over power blocks
            push!(sector_p_df, [tii, dev, sector_no, sector_p])
            sector_no += 1
        end

        end
    end


    # Writing the ts parameters into a h5 file
    h5open(ts_params_file, "w") do file
        file["duration/time_index"] = duration_df.time_index
        file["duration/duration"] = duration_df.duration

        file["P_lims/time_index"] = P_ulb_df.time_index
        file["P_lims/dev"] = P_ulb_df.dev
        file["P_lims/p_lb"] = P_ulb_df.p_lb
        file["P_lims/p_ub"] = P_ulb_df.p_ub
        
        file["sector_costs/time_index"] = sector_cst_df.time_index
        file["sector_costs/dev"] = sector_cst_df.dev
        file["sector_costs/sector_no"] = sector_cst_df.sector_no
        file["sector_costs/sector_cst"] = sector_cst_df.sector_cst

        file["sector_powers/time_index"] = sector_p_df.time_index
        file["sector_powers/dev"] = sector_p_df.dev
        file["sector_powers/sector_no"] = sector_p_df.sector_no
        file["sector_powers/sector_p"] = sector_p_df.sector_p
    end
end


# identify the data
# I am passing the file path and the created file names as an argument, looping over them here
network_file_name = ARGS[1]
static_params_name = ARGS[2]
ts_params_name = ARGS[3]
exportNetworkFiles(network_file_name, static_params_name, ts_params_name)

