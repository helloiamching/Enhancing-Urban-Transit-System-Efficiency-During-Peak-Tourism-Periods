# -*- coding: utf-8 -*-
'''
'''
import torch

def edge():
    pe_stations = [
    'PE1', 'PE2', 'PE3', 'PE4', 'PE5', 'PE6', 'PE7'
    ]

    pw_stations = [
        'PW1', 'PW2', 'PW3', 'PW4', 'PW5', 'PW6', 'PW7'   
    ]

    se_stations = [
        'SE1', 'SE2', 'SE3','SE4', 'SE5'
    ]

    sw_stations = [
        'SW1', 'SW2', 'SW3', 'SW4', 'SW5', 'SW6', 'SW7', 'SW8'
    ]
    bp_stations = [
        'BP2', 'BP3', 'BP4', 'BP5', 'BP6/DT1', 'BP7',
        'BP8', 'BP9', 'BP10', 'BP11', 'BP12', 'BP13'
    ]
      
    cc_stations = [
        'CE1/DT16', 'NS24/NE6/CC1', 'CC2', 'CC3', 'CC4/DT15', 'CC5', 'CC6', 'CC7', 'CC8', 'EW8/CC9', 'CC10/DT26',
        'CC11', 'CC12', 'NE12/CC13', 'CC14', 'NS17/CC15', 'CC16', 'CC17/TE9', 'CC19/DT9', 'CC20',
        'CC21', 'EW21/CC22', 'CC23', 'CC24', 'CC25', 'CC26', 'CC27', 'CC28', 'NE1/CC29'
    ]
    dt_stations = [
        'BP6/DT1', 'DT2', 'DT3', 'DT5', 'DT6', 'DT7', 'DT8', 'CC19/DT9', 
        'DT10', 'NS21/DT11', 'NE7/DT12', 'DT13', 'EW12/DT14', 'CC4/DT15', 'CE1/DT16', 'DT17',
        'DT18', 'NE4/DT19', 'DT20', 'DT21', 'DT22', 'DT23', 'DT24', 'DT25', 'CC10/DT26', 
        'DT27', 'DT28', 'DT29', 'DT30', 'DT31', 'EW2/DT32', 'DT33', 'DT34', 'CG1/DT35'
    ]
    
    cg_stations = [
      'CG1/DT35', 'CG2'
    ]
    ew_stations = [
        'EW1', 'EW2/DT32', 'EW3', 'EW4', 'EW5', 'EW6', 'EW7', 'EW8/CC9', 'EW9', 'EW10',
        'EW11', 'EW12/DT14', 'NS25/EW13', 'EW14/NS26', 'EW15', 'EW16/NE3/TE17', 'EW17', 'EW18',
        'EW19', 'EW20', 'EW21/CC22', 'EW22', 'EW23', 'EW24/NS1', 'EW25', 'EW26',
        'EW27', 'EW28', 'EW29', 'EW30', 'EW31', 'EW32', 'EW33'
    ]
    
    ne_stations = [
        'NE1/CC29', 'EW16/NE3/TE17', 'NE4/DT19', 'NE5', 'NS24/NE6/CC1', 'NE7/DT12', 
        'NE8', 'NE9', 'NE10', 'NE11', 'NE12/CC13', 'NE13', 'NE14', 'NE15', 'NE16/STC', 
        'NE17/PTC', 'NE18'
    ]
    
    ns_stations = [
        'EW24/NS1', 'NS2', 'NS3', 'NS4/BP1', 'NS5', 'NS7', 'NS8', 'NS9/TE2', 'NS10', 'NS11',
        'NS12', 'NS13', 'NS14', 'NS15', 'NS16', 'NS17/CC15', 'NS18', 'NS19', 'NS20', 
        'NS21/DT11', 'TE14/NS22', 'NS23', 'NS24/NE6/CC1', 'NS25/EW13', 'EW14/NS26', 'NS27/CE2/TE20', 'NS28'
    ]
    
    te_stations = [
        'TE1', 'NS9/TE2', 'TE3', 'TE4', 'TE5', 'TE6', 'TE7', 'TE8', 'CC17/TE9', 
        'TE11', 'TE12', 'TE13', 'TE14/NS22', 'TE15', 'TE16', 'EW16/NE3/TE17',
        'TE18', 'TE19', 'NS27/CE2/TE20', 'TE22', 'TE23', 'TE24',
        'TE25', 'TE26', 'TE27', 'TE28', 'TE29'
     ]
    
    # 合并所有站点
    all_stations = list(dict.fromkeys(bp_stations + cc_stations + dt_stations + ew_stations 
                                      + ne_stations + ns_stations + te_stations + cg_stations 
                                      + pe_stations + pw_stations + se_stations + sw_stations))
    
    # 为所有站点统一分配索引
    station_index = {name: i for i, name in enumerate(all_stations)}
    
    # 边构造函数（从站点序列生成无向边）
    def build_edges(station_list):
        edges = []
        for i in range(len(station_list) - 1):
            a = station_index[station_list[i]]
            b = station_index[station_list[i + 1]]
            edges.append((a, b))  # 正向边
            edges.append((b, a))  # 反向边
        return edges
    
    # 构造两条线的边
    edges_bp = build_edges(bp_stations)
    edges_cc = build_edges(cc_stations)
    edges_dt = build_edges(dt_stations)
    edges_ew = build_edges(ew_stations)
    edges_ne = build_edges(ne_stations)
    edges_ns = build_edges(ns_stations)
    edges_te = build_edges(te_stations)
    edges_cg = build_edges(cg_stations)
    edges_pe = build_edges(pe_stations)
    edges_pw = build_edges(pw_stations)
    edges_se = build_edges(se_stations)
    edges_sw = build_edges(sw_stations)

    
    
    # 合并边并转为 PyG 格式 edge_index
    all_edges = edges_bp + edges_cc + edges_dt + edges_ew + edges_ne + edges_ns + edges_te + edges_cg + edges_pe + edges_pw + edges_se + edges_sw
    edge_index = torch.tensor(all_edges, dtype=torch.long).t()  # shape: [2, num_edges]
    
    return edge_index, station_index



 
