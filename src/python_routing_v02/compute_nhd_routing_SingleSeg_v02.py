#!/usr/bin/env python
# coding: utf-8
# example usage: python compute_nhd_routing_SingleSeg.py -v -t -w -n Mainstems_CONUS


# -*- coding: utf-8 -*-
"""NHD Network traversal

A demonstration version of this code is stored in this Colaboratory notebook:
    https://colab.research.google.com/drive/1ocgg1JiOGBUl3jfSUPCEVnW5WNaqLKCD

"""
## Parallel execution
import os
import sys
sys.setrecursionlimit(30000)
import time
import numpy as np
import argparse
import pathlib
import pandas as pd
from functools import partial
from joblib import delayed, Parallel
from itertools import chain, islice
from operator import itemgetter


def _handle_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--debuglevel",
        help="Set the debuglevel",
        dest="debuglevel",
        choices=[0, -1, -2, -3],
        default=0,
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose output (leave blank for quiet output)",
        dest="verbose",
        action="store_true",
    )
    parser.add_argument(
        "--assume-short-ts",
        help="Use the previous timestep value for upstream flow",
        dest="assume_short_ts",
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write output files (leave blank for no writing)",
        dest="write_output",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--showtiming",
        help="Set the showtiming (leave blank for no timing information)",
        dest="showtiming",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--break-at-waterbodies",
        help="Use the waterbodies in the route-link dataset to divide the computation (leave blank for no splitting)",
        dest="break_network_at_waterbodies",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--supernetwork",
        help="Choose from among the pre-programmed supernetworks (Pocono_TEST1, Pocono_TEST2, LowerColorado_Conchos_FULL_RES, Brazos_LowerColorado_ge5, Brazos_LowerColorado_FULL_RES, Brazos_LowerColorado_Named_Streams, CONUS_ge5, Mainstems_CONUS, CONUS_Named_Streams, CONUS_FULL_RES_v20",
        choices=[
            "Pocono_TEST1",
            "Pocono_TEST2",
            "LowerColorado_Conchos_FULL_RES",
            "Brazos_LowerColorado_ge5",
            "Brazos_LowerColorado_FULL_RES",
            "Brazos_LowerColorado_Named_Streams",
            "CONUS_ge5",
            "Mainstems_CONUS",
            "CONUS_Named_Streams",
            "CONUS_FULL_RES_v20",
        ],
        # TODO: accept multiple or a Path (argparse Action perhaps)
        # action='append',
        # nargs=1,
        dest="supernetwork",
        default="Pocono_TEST1",
    )
    parser.add_argument("--ql", help="QLat input data", dest="ql", default=None)

    return parser.parse_args()


ENV_IS_CL = False
if ENV_IS_CL:
    root = pathlib.Path("/", "content", "t-route")
elif not ENV_IS_CL:
    root = pathlib.Path("../..").resolve()
    #sys.path.append(r"../python_framework_v02")

    # TODO: automate compile for the package scripts
    sys.path.append("fast_reach")

## network and reach utilities
import troute.nhd_network_utilities_v02 as nnu
import mc_reach
import troute.nhd_network as nhd_network
import troute.nhd_io as nhd_io


def writetoFile(file, writeString):
    file.write(writeString)
    file.write("\n")


def constant_qlats(data, nsteps, qlat):
    q = np.full((len(data.index), nsteps), qlat, dtype="float32")
    ql = pd.DataFrame(q, index=data.index, columns=range(nsteps))
    return ql


def main():

    args = _handle_args()

    nts = 144
    debuglevel = -1 * args.debuglevel
    verbose = args.verbose
    showtiming = args.showtiming
    supernetwork = args.supernetwork
    break_network_at_waterbodies = args.break_network_at_waterbodies
    write_output = args.write_output
    assume_short_ts = args.assume_short_ts

    test_folder = pathlib.Path(root, "test")
    geo_input_folder = test_folder.joinpath("input", "geo")

    # TODO: Make these commandline args
    """##NHD Subset (Brazos/Lower Colorado)"""
    # supernetwork = 'Brazos_LowerColorado_Named_Streams'
    # supernetwork = 'Brazos_LowerColorado_ge5'
    # supernetwork = 'Pocono_TEST1'
    """##NHD CONUS order 5 and greater"""
    # supernetwork = 'CONUS_ge5'
    """These are large -- be careful"""
    # supernetwork = 'Mainstems_CONUS'
    # supernetwork = 'CONUS_FULL_RES_v20'
    # supernetwork = 'CONUS_Named_Streams' #create a subset of the full resolution by reading the GNIS field
    # supernetwork = 'CONUS_Named_combined' #process the Named streams through the Full-Res paths to join the many hanging reaches

    if verbose:
        print("creating supernetwork connections set")
    if showtiming:
        start_time = time.time()

    # STEP 1
    network_data = nnu.set_supernetwork_data(
        supernetwork=args.supernetwork,
        geo_input_folder=geo_input_folder,
        verbose=False,
        debuglevel=debuglevel,
    )

    cols = network_data["columns"]
    data = nhd_io.read(network_data["geo_file_path"])
    data = data[list(cols.values())]
    data = data.set_index(cols["key"])

    if "mask_file_path" in network_data:
        data_mask = nhd_io.read_mask(
            network_data["mask_file_path"],
            layer_string=network_data["mask_layer_string"],
        )
        data = data.filter(data_mask.iloc[:, network_data["mask_key"]], axis=0)

    data = nhd_io.replace_downstreams(data, cols["downstream"], 0)

    """
    print(data.head())
    print(network_data['columns'])
    import os
    os._exit(1)
    """
    if args.ql:
        qlats = nhd_io.read_qlat(args.ql)
    else:
        qlats = constant_qlats(data, nts, 10.0)

    # initial conditions, assume to be zero
    # TO DO: Allow optional reading of initial conditions from WRF
    q0 = pd.DataFrame(0,index = data.index, columns = ["qu0","qd0","h0"], dtype = "float32")
    print(data.head())
    connections = nhd_network.extract_connections(data, cols["downstream"])
    #connections is just idx -> to/downstream in data
    print(data.head())

    wbodies = nhd_network.extract_waterbodies(
        data, cols["waterbody"], network_data["waterbody_null_code"]
    )

    if verbose:
        print("supernetwork connections set complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    # STEP 2
    if showtiming:
        start_time = time.time()
    if verbose:
        print("organizing connections into reaches ...")

    def reverse_break_reaches(data):
        # rconn is just inverted data, i.e. idx -> from
        conn = data['to'].abs()
        # to is many to one
        rconn = conn.reset_index().set_index('to')['link']
        # init the reverse map with a negative identity
        data['from'] = -1*data.index
        # for any reversible entries, overwrite with id of upstram
        # rconn does NOT have a uniquie index
        # to break up into "reaches" simply drop anything with duplicate indicies
        # which implies more than one connection
        reach_idx = rconn.index.drop_duplicates(keep=False)
        data.loc[reach_idx, 'from'] = rconn.loc[reach_idx]


    def niave_reversal(data):
        #rconn is just inverted data, i.e. idx -> from
        conn = data['to'].abs()
        # to is many to one
        rconn = conn.reset_index().set_index('to')['link']
        rconn.name = 'from'
        #init the reverse map with a negative identity
        data['from'] = -1*data.index
        #for any reversible entries, overwrite with id of upstram
        #rconn does NOT have a uniquie index
        # this niave approach picks (somewhat arbitratily) one of the possible upstreams
        data.loc[rconn.index, 'from'] = rconn

    def get_connections(data):
        # Get all connections forming a multi index
        # Prepare a index of 'to' ids
        conn = data['to'][ data['to'] > 0 ]
        rconn = conn.reset_index().set_index('to')['link']
        # this is now the inverse mapping, possible many-1
        rconn.name = 'from'
        # really on need the ids to create the connections
        connections = data.reset_index()[['link', 'to']]
        # merge the two indicies, will provide duplicates in 'link' but have a new reset index
        connections = pd.merge(connections, rconn, how='outer', left_on='link', right_index=True)
        # where the mapping fails (i.e. there is no from to map) there will be NaN
        # fill this in with the -id value
        connections['from'].fillna( connections['link']*-1, inplace=True )
        # make sure the values are proper ints
        connections['from'] = connections['from'].astype('int64')

        return connections

    test = 948060164
    print(data.loc[ [test] ])

    connections = get_connections(data)
    # drop the to before merging, don't need two of them
    connections = connections.drop('to', axis=1)
    data = data.reset_index().merge(connections, how='outer', right_on='link', left_on='link')
    #Create the multi-index
    data.set_index(['link', 'from'], inplace=True)

    """
    sub = data.loc[ test ]
    rconn_sub = rconn.loc[ test ]

    print("Access non-unique index value")
    print(sub)
    print("rconn_sub")
    print(rconn_sub)
    os._exit(1)
    """
    #FIXME rename reachable_network to network_id
    data['reachable_network'] = -1
    network_id = 0
    #print(data.head())
    def mark_reachable(id, network_id):

        data.loc[id, 'reachable_network'] = network_id
        #mark the upstream
        if data.loc[id]['from'] > 0:
            mark_reachable(data.loc[id]['from'], network_id)
        #mark the downstream
        if data.loc[id]['to'] > 0:
            mark_reachable(data.loc[id]['to'], network_id)

        def mark_network_id(s, network_id):
            if s['reachable_network'] == -1:
                network_id = network_id + 1
                nid = network_id
            else:
                nid = s['reachable_network']
            data.loc[id, 'reachable_network'] = nid
            if data.loc[s.name]['from'] > 0:
                data.loc[ data.loc[s.name]['from'], 'reachable_network'] = nid
            if data.loc[s.name]['to'] > 0:
                data.loc[ data.loc[s.name]['to'], 'reachable_network'] = nid
    #data.apply(mark_network_id, axis=1, args=(network_id,))

    networks = {}
    networks[network_id] = set()
    import time
    t0 = time.time()

    for row in data[['to', 'from', 'reachable_network']].itertuples():
        #for some weird reason, the named tuple gives the from value as _2
        #print(row)
        #os._exit(1)
        #if row.reachable_network == -1:
        #    network_id = network_id + 1
        #    networks[network_id] = []
        #    nid = network_id
        #else:
        #    nid = row.reachable_network
        id = row.Index
        nid = -1
        for k, network in networks.items():
            if id in network:
                nid = k
        if nid == -1:
            network_id = network_id+1
            networks[network_id] = set()
            nid = network_id
        networks[nid].add(id)
        #data.loc[id, 'reachable_network'] = nid
        if row._2 > 0:
            #data.loc[row._2, 'reachable_network'] = nid
            networks[nid].add( row._2 )
        if row.to > 0:
            #data.loc[row.to, 'reachable_network'] = nid
            networks[nid].add( row.to )

    for network_id, network in networks.items():
        data.loc[network, 'reachable_network'] = network_id
    # Cut the large, independent networks into sub-networks
    # can break at "arbitrary" points, heuristic based for best
    # distribution of compututation (BFS?)
    t1 = time.time()
    print('reachable')
    print(data.loc[ [test] ])
    os._exit(1)
    """
    print("itertuples in :", t1-t0)
    print(networks[1])
    print(network_id)
    print(networks[0])
    print(data.head())
    os._exit(1)
    """
    """
    for row_idx in data.index:
        row = data.loc[row_idx]

        if row.reachable_network == -1:
            network_id = network_id + 1
            #networks[network_id] = []
            nid = network_id
            #networks[nid].add(id)
        data.loc[row_idx, 'reachable_network'] = nid
        if row['from'] > 0:
            data.loc[row['from'], 'reachable_network'] = nid
            #networks[nid].add( row._2 )
        if row.to > 0:
            data.loc[row.to, 'reachable_network'] = nid
            #networks[nid].add( row.to )
    t1 = time.time()
    print("itertuples in :", t1-t0)
    #print(networks[1])
    print(network_id)
    #print(networks[0])
    os._exit(1)
    """

    """
    for id, row in data[['to', 'from', 'reachable_network']].iterrows():
        if row['reachable_network'] == -1:
            network_id = network_id + 1
            nid = network_id
        else:
            nid = row['reachable_network']
    """
    """
        data.loc[id, 'reachable_network'] = nid
        if data.loc[id]['from'] > 0:
            data.loc[ data.loc[id]['from'], 'reachable_network'] = nid
        if data.loc[id]['to'] > 0:
            data.loc[ data.loc[id]['to'], 'reachable_network'] = nid
    """
    #print(data.head())

    rconn = nhd_network.reverse_network(connections)
    #*POTENTIALLY NON DENDRIDIC* partitions of the supernework
    subnets = nhd_network.reachable_network(rconn)
    #print(len(subnets.keys()))

    print( data.groupby('reachable_network').get_group(1))
    print(data.loc[7065882][['from', 'to', 'reachable_network']])
    print(data.groupby('reachable_network').get_group(21))

    print( data[ ['to', 'from', 'Length', 'reachable_network'] ] )
    print( data['reachable_network'].max() )
    #*DENDRIDIC* sets of ORDERED reaches (subnetwork)
    subreaches = {}
    for tw, net in subnets.items():
        path_func = partial(nhd_network.split_at_junction, net)
        subreaches[tw] = nhd_network.dfs_decomposition(net, path_func)
    print(len(subreaches.keys()))
    os._exit(1)
    #import os
    #print(subreaches)
    #os._exit(1)
    if verbose:
        print("reach organization complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    if showtiming:
        start_time = time.time()

    data["dt"] = 300.0
    data = data.rename(columns=nnu.reverse_dict(cols))
    data = data.astype("float32")

    # datasub = data[['dt', 'bw', 'tw', 'twcc', 'dx', 'n', 'ncc', 'cs', 's0']]
    import cProfile
    pr = cProfile.Profile()

    parallelcompute = False
    if parallelcompute:
        with Parallel(n_jobs=-1, backend="threading") as parallel:
            jobs = []
            for twi, (tw, reach) in enumerate(subreaches.items(), 1):
                r = list(chain.from_iterable(reach))
                data_sub = data.loc[
                    r, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
                ].sort_index()
                qlat_sub = qlats.loc[r].sort_index()
                q0_sub = q0.loc[r].sort_index()
                jobs.append(
                    delayed(mc_reach.compute_network)(
                        nts,
                        reach,
                        subnets[tw],
                        data_sub.index.values,
                        data_sub.columns.values,
                        data_sub.values,
                        qlat_sub.values,
                        q0_sub.values,
                    )
                )
            results = parallel(jobs)
    else:
        results = []
        for twi, (tw, reach) in enumerate(subreaches.items(), 1):
            r = list(chain.from_iterable(reach))
            #data_sub = data.loc[
            #    r, ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0"]
            #    ].sort_index()
            data_sub = data.loc[
                r, ["dt", "dx", "bw", "tw", "twcc", "n", "ncc", "cs", "s0"]
                ].sort_index()
            #print(data_sub.head())
            qlat_sub = qlats.loc[r].sort_index()
            q0_sub = q0.loc[r].sort_index()
            #q0_sub['qu0'] = q0_sub.index.values.astype("float32")
            """
            print("Reach")
            print(reach)
            print("Subnets")
            print(subnets[tw])
            print("data")
            print(data_sub)
            print("qlat")
            print(qlat_sub)
            print("init")
            print(q0_sub)
            os._exit(1)
            """
            #if(len(reach) > 1):
            #    print("HERE")
            #    print(reach)
            #    print("HERE AGAIN")
            #    print(subnets[tw])
            #    print("IDX")
            #    print(data_sub.index)
            #    print(data_sub)
            #pr.enable()
            #mc_reach.compute_networks(nts, subnets)
            results.append(
                mc_reach.compute_network_structured_obj(
                #mc_reach.compute_network(
                    nts,
                    reach,
                    subnets[tw],
                    data_sub.index.values,
                    data_sub.columns.values,
                    data_sub.values,
                    qlat_sub.values,
                    q0_sub.values,
                )

            )
            #pr.disable()

    fdv_columns = pd.MultiIndex.from_product(
        [range(nts), ["q", "v", "d"]]
    ).to_flat_index()
    flowveldepth = pd.concat(
        [pd.DataFrame(d, index=i, columns=fdv_columns) for i, d in results], copy=False
    )
    flowveldepth = flowveldepth.sort_index()
    #flowveldepth.to_csv(f"{args.supernetwork}.csv")
    #print(flowveldepth)

    if verbose:
        print("ordered reach computation complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    #pr.print_stats(sort='time')
if __name__ == "__main__":
    main()
