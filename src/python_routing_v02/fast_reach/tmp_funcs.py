
cpdef object compute_network_structured(int nsteps, list reaches, dict connections,
    const long[:] data_idx, object[:] data_cols, const float[:,:] data_values,
    const float[:, :] qlat_values, const float[:,:] initial_conditions,
    # const float[:] wbody_idx, object[:] wbody_cols, const float[:, :] wbody_vals,
    bint assume_short_ts=False):
    """
    Compute network
    Args:
        nsteps (int): number of time steps
        reaches (list): List of reaches
        connections (dict): Network
        data_idx (ndarray): a 1D sorted index for data_values
        data_values (ndarray): a 2D array of data inputs (nodes x variables)
        qlats (ndarray): a 2D array of qlat values (nodes x nsteps). The index must be shared with data_values
        initial_conditions (ndarray): an n x 3 array of initial conditions. n = nodes, column 1 = qu0, column 2 = qd0, column 3 = h0
        assume_short_ts (bool): Assume short time steps (quc = qup)
    Notes:
        Array dimensions are checked as a precondition to this method.
    """
    # Check shapes
    if qlat_values.shape[0] != data_idx.shape[0]:
        raise ValueError(f"Number of rows in Qlat is incorrect: expected ({data_idx.shape[0]}), got ({qlat_values.shape[0]})")
    if qlat_values.shape[1] > nsteps:
        raise ValueError(f"Number of columns (timesteps) in Qlat is incorrect: expected at most ({data_idx.shape[0]}), got ({qlat_values.shape[0]}). The number of columns in Qlat must be equal to or less than the number of routing timesteps")
    if data_values.shape[0] != data_idx.shape[0] or data_values.shape[1] != data_cols.shape[0]:
        raise ValueError(f"data_values shape mismatch")
    #define an intialize the final output array
    cdef np.ndarray[float, ndim=3] flowveldepth = np.zeros((data_idx.shape[0], nsteps+1, 3), dtype='float32')
    #Make ndarrays from the mem views for convience of indexing...may be a better method
    cdef np.ndarray[float, ndim=2] data_array = np.asarray(data_values)
    cdef np.ndarray[float, ndim=2] init_array = np.asarray(initial_conditions)
    cdef np.ndarray[float, ndim=2] qlat_array = np.asarray(qlat_values)
    ###### Declare/type variables #####
    cdef Py_ssize_t max_buff_size = 0
    #lists to hold reach definitions, i.e. list of ids
    cdef list reach
    cdef list upstream_reach
    #lists to hold segment ids
    cdef list segment_ids
    cdef list upstream_ids
    #flow accumulation variables
    cdef float upstream_flows, previous_upstream_flows
    #starting timestep, shifted by 1 to account for initial conditions
    cdef int timestep = 1
    #buffers to pass to compute_reach_kernel
    cdef float[:,:] buf_view
    cdef float[:,:] out_buf
    cdef float[:] lateral_flows
    #reach iterator
    cdef MC_Reach r
    # list of reach objects to operate on
    cdef list reach_objects = []
    cdef list segment_objects
    #pre compute the qlat resample fraction
    cdef double qlat_resample = nsteps/qlat_values.shape[1]

    #Preprocess the raw reaches, creating MC_Reach/MC_Segments
    for reach in reaches:
      upstream_reach = connections.get(reach[0], ())
      #reach_len = len(reach)
      #upstream_reach_len = len(upstream_reach)
      segment_ids = binary_find(data_idx, reach)
      upstream_ids = binary_find(data_idx, upstream_reach)

      #params = data_array[segment_ids]
      #inits = init_array[segment_ids]
      #Set the initial condtions before running loop
      flowveldepth[segment_ids, 0] = init_array[segment_ids]
      #qlats = qlat_array[segment_ids]

      segment_objects = []
      #Find the max reach size, used to create buffer for compute_reach_kernel
      if len(segment_ids) > max_buff_size:
        max_buff_size=len(segment_ids)

      for segment in segment_ids:
        #FIXME data_array order is important, column_mapper might help with this
        segment_objects.append(
            MC_Segment(segment, *data_array[segment], *init_array[segment])
            )

      reach_objects.append(
          MC_Reach(segment_objects, array('l',upstream_ids))
          )

    #Init buffers
    lateral_flows = np.zeros( max_buff_size, dtype='float32' )
    buf_view = np.zeros( (max_buff_size, 13), dtype='float32')
    out_buf = np.full( (max_buff_size, 3), -1, dtype='float32')

    #Run time
    while timestep < nsteps:
      for r in reach_objects:
            #Need to get quc and qup
            upstream_flows = 0.0
            previous_upstream_flows = 0.0
            for id in r.upstream_ids: #Explicit loop reduces some overhead
              upstream_flows += flowveldepth[id, timestep, 0]
              previous_upstream_flows += flowveldepth[id, timestep-1, 0]
            #Index of segments required to process this reach
            segment_ids = [segment.id for segment in r.segments]

            if assume_short_ts:
                upstream_flows = previous_upstream_flows
            #Create compute reach kernel input buffer
            for i, segment in enumerate(r.segments):
              buf_view[i][0] = qlat_array[ segment.id, int(timestep/qlat_resample)]
              buf_view[i][1] = segment.dt
              buf_view[i][2] = segment.dx
              buf_view[i][3] = segment.bw
              buf_view[i][4] = segment.tw
              buf_view[i][5] = segment.twcc
              buf_view[i][6] = segment.n
              buf_view[i][7] = segment.ncc
              buf_view[i][8] = segment.cs
              buf_view[i][9] = segment.s0
              buf_view[i][10] = flowveldepth[segment.id, timestep-1, 0]
              buf_view[i][11] = flowveldepth[segment.id, timestep-1, 1]
              buf_view[i][12] = flowveldepth[segment.id, timestep-1, 2]

            compute_reach_kernel(previous_upstream_flows, upstream_flows,
                                 len(r.segments), buf_view,
                                 out_buf,
                                 assume_short_ts)
            #Copy the output out
            for i, id in enumerate(segment_ids):
              flowveldepth[id, timestep-1, 0] = out_buf[i, 0]
              flowveldepth[id, timestep-1, 1] = out_buf[i, 1]
              flowveldepth[id, timestep-1, 2] = out_buf[i, 2]

      timestep += 1
    #drop initial condition before returning
    flowveldepth = flowveldepth[:,1:,:]
    return np.asarray(data_idx, dtype=np.intp), np.asarray(flowveldepth.reshape(flowveldepth.shape[0], -1), dtype='float32')

cpdef object compute_network_groups(int nsteps, list reaches, dict connections,
    const long[:] data_idx, object[:] data_cols, const float[:,:] data_values,
    const float[:, :] qlat_values, const float[:,:] initial_conditions,
    # const float[:] wbody_idx, object[:] wbody_cols, const float[:, :] wbody_vals,
    bint assume_short_ts=False):
    """
    Compute network
    Args:
        nsteps (int): number of time steps
        reaches (list): List of reaches
        connections (dict): Network
        data_idx (ndarray): a 1D sorted index for data_values
        data_values (ndarray): a 2D array of data inputs (nodes x variables)
        qlats (ndarray): a 2D array of qlat values (nodes x nsteps). The index must be shared with data_values
        initial_conditions (ndarray): an n x 3 array of initial conditions. n = nodes, column 1 = qu0, column 2 = qd0, column 3 = h0
        assume_short_ts (bool): Assume short time steps (quc = qup)
    Notes:
        Array dimensions are checked as a precondition to this method.
    """
    # Check shapes
    if qlat_values.shape[0] != data_idx.shape[0]:
        raise ValueError(f"Number of rows in Qlat is incorrect: expected ({data_idx.shape[0]}), got ({qlat_values.shape[0]})")
    if qlat_values.shape[1] > nsteps:
        raise ValueError(f"Number of columns (timesteps) in Qlat is incorrect: expected at most ({data_idx.shape[0]}), got ({qlat_values.shape[0]}). The number of columns in Qlat must be equal to or less than the number of routing timesteps")
    if data_values.shape[0] != data_idx.shape[0] or data_values.shape[1] != data_cols.shape[0]:
        raise ValueError(f"data_values shape mismatch")
    #define an intialize the final output array
    cdef np.ndarray[float, ndim=3] flowveldepth = np.zeros((data_idx.shape[0], nsteps, 3), dtype='float32')
    #Make ndarrays from the mem views for convience of indexing...may be a better method
    cdef np.ndarray[float, ndim=2] data_array = np.asarray(data_values)
    cdef np.ndarray[float, ndim=2] init_array = np.asarray(initial_conditions)
    cdef np.ndarray[float, ndim=2] qlat_array = np.asarray(qlat_values)
    ###### Declare/type variables #####
    cdef Py_ssize_t max_buff_size = 0
    #lists to hold reach definitions, i.e. list of ids
    cdef list reach
    cdef list upstream_reach
    #lists to hold segment ids
    cdef list segment_ids
    cdef list upstream_ids
    #flow accumulation variables
    cdef float upstream_flows, previous_upstream_flows
    #starting timestep, shifted by 1 to account for initial conditions
    cdef int timestep = 1
    #buffers to pass to compute_reach_kernel
    cdef float[:,:] buf_view
    cdef float[:,:] out_buf
    cdef float[:] lateral_flows
    #reach iterator
    #cdef MC_Reach r
    # list of reach objects to operate on
    cdef list reach_objects = []
    cdef list segment_objects
    #pre compute the qlat resample fraction
    cdef double qlat_resample = nsteps/qlat_values.shape[1]

    #Preprocess the raw reaches, creating MC_Reach/MC_Segments
    for reach in reaches:
      upstream_reach = connections.get(reach[0], ())
      #reach_len = len(reach)
      #upstream_reach_len = len(upstream_reach)
      segment_ids = binary_find(data_idx, reach)
      upstream_ids = binary_find(data_idx, upstream_reach)

      #params = data_array[segment_ids]
      #inits = init_array[segment_ids]
      #Set the initial condtions before running loop
      flowveldepth[segment_ids, 0] = init_array[segment_ids]
      #qlats = qlat_array[segment_ids]

      segment_objects = []
      #Find the max reach size, used to create buffer for compute_reach_kernel
      if len(segment_ids) > max_buff_size:
        max_buff_size=len(segment_ids)

      for segment in segment_ids:
        #FIXME data_array order is important, column_mapper might help with this
        segment_objects.append(
            MC_Segment(segment, *data_array[segment], *init_array[segment])
            )

      reach_objects.append(
          MC_Reach(segment_objects, array('l',upstream_ids))
          )

    #Init buffers
    lateral_flows = np.zeros( max_buff_size, dtype='float32' )
    buf_view = np.zeros( (max_buff_size, 13), dtype='float32')
    out_buf = np.full( (max_buff_size, 3), -1, dtype='float32')

    #Run time
    while timestep < nsteps:
      for r in reach_objects:
            #Need to get quc and qup
            upstream_flows = 0.0
            previous_upstream_flows = 0.0
            for id in r.upstream_ids: #Explicit loop reduces some overhead
              upstream_flows += flowveldepth[id, timestep, 0]
              previous_upstream_flows += flowveldepth[id, timestep-1, 0]
            #Index of segments required to process this reach
            segment_ids = [segment.id for segment in r.segments]

            if assume_short_ts:
                upstream_flows = previous_upstream_flows
            #Create compute reach kernel input buffer
            for i, segment in enumerate(r.segments):
              buf_view[i][0] = qlat_array[ segment.id, int(timestep/(nsteps/qlat_values.shape[1]))]
              buf_view[i][1] = segment.dt
              buf_view[i][2] = segment.dx
              buf_view[i][3] = segment.bw
              buf_view[i][4] = segment.tw
              buf_view[i][5] = segment.twcc
              buf_view[i][6] = segment.n
              buf_view[i][7] = segment.ncc
              buf_view[i][8] = segment.cs
              buf_view[i][9] = segment.s0
              buf_view[i][10] = flowveldepth[segment.id, timestep-1, 0]
              buf_view[i][11] = flowveldepth[segment.id, timestep-1, 1]
              buf_view[i][12] = flowveldepth[segment.id, timestep-1, 2]

            compute_reach_kernel(previous_upstream_flows, upstream_flows,
                                 len(r.segments), buf_view,
                                 out_buf,
                                 assume_short_ts)
            #Copy the output out
            for i, id in enumerate(segment_ids):
              flowveldepth[id, timestep, 0] = out_buf[i, 0]
              flowveldepth[id, timestep, 1] = out_buf[i, 1]
              flowveldepth[id, timestep, 2] = out_buf[i, 2]

      timestep += 1

    return np.asarray(data_idx, dtype=np.intp), np.asarray(flowveldepth.reshape(flowveldepth.shape[0], -1), dtype='float32')








#####WORKING EXCEPT FOR SLICING OFF INIT condition
cpdef object compute_network_groups(int nsteps, list reaches, dict connections,
    const long[:] data_idx, object[:] data_cols, const float[:,:] data_values,
    const float[:, :] qlat_values, const float[:,:] initial_conditions,
    # const float[:] wbody_idx, object[:] wbody_cols, const float[:, :] wbody_vals,
    bint assume_short_ts=False):
    """
    Compute network
    Args:
        nsteps (int): number of time steps
        reaches (list): List of reaches
        connections (dict): Network
        data_idx (ndarray): a 1D sorted index for data_values
        data_values (ndarray): a 2D array of data inputs (nodes x variables)
        qlats (ndarray): a 2D array of qlat values (nodes x nsteps). The index must be shared with data_values
        initial_conditions (ndarray): an n x 3 array of initial conditions. n = nodes, column 1 = qu0, column 2 = qd0, column 3 = h0
        assume_short_ts (bool): Assume short time steps (quc = qup)
    Notes:
        Array dimensions are checked as a precondition to this method.
    """
    # Check shapes
    if qlat_values.shape[0] != data_idx.shape[0]:
        raise ValueError(f"Number of rows in Qlat is incorrect: expected ({data_idx.shape[0]}), got ({qlat_values.shape[0]})")
    if qlat_values.shape[1] > nsteps:
        raise ValueError(f"Number of columns (timesteps) in Qlat is incorrect: expected at most ({data_idx.shape[0]}), got ({qlat_values.shape[0]}). The number of columns in Qlat must be equal to or less than the number of routing timesteps")
    if data_values.shape[0] != data_idx.shape[0] or data_values.shape[1] != data_cols.shape[0]:
        raise ValueError(f"data_values shape mismatch")

    cdef np.ndarray[float, ndim=3] flowveldepth = np.zeros((data_idx.shape[0], nsteps, 3), dtype='float32')
    # copy reaches into an array
    cdef list reach_objects = []
    #Make ndarrays from the mem views for convience of indexing...may be a better method
    cdef np.ndarray[float, ndim=2] data_array = np.asarray(data_values)
    cdef np.ndarray[float, ndim=2] init_array = np.asarray(initial_conditions)
    cdef np.ndarray[float, ndim=2] qlat_array = np.asarray(qlat_values)
    cdef Py_ssize_t max_buff_size = 0

    for reach in reaches:
      upstream_reach = connections.get(reach[0], ())
      reach_len = len(reach)
      upstream_reach_len = len(upstream_reach)
      segment_ids = binary_find(data_idx, reach)
      upstream_ids = binary_find(data_idx, upstream_reach)

      params = data_array[segment_ids]
      inits = init_array[segment_ids]
      #Set the initial condtions before running loop
      flowveldepth[segment_ids, 0] = inits
      #qlats = qlat_array[segment_ids]

      segment_objects = []
      #Find the max reach size, used to create buffer for compute_reach_kernel
      if len(segment_ids) > max_buff_size:
        max_buff_size=len(segment_ids)

      for segment in segment_ids:
        #FIXME data_array order is important, column_mapper might help with this
        segment_objects.append(
            MC_Segment(segment, *data_array[segment], *init_array[segment])
            )

      reach_objects.append(
          MC_Reach(segment_objects, array('l',upstream_ids))
          )

    cdef float upstream_flows, previous_upstream_flows
    cdef int timestep = 0

    cdef float[:,:] buf_view
    cdef float[:,:] out_buf
    cdef float[:] lateral_flows
    #Init buffers
    lateral_flows = np.zeros( max_buff_size, dtype='float32' )
    buf_view = np.zeros( (max_buff_size, 13), dtype='float32')
    out_buf = np.full( (max_buff_size, 3), -1, dtype='float32')
    #Run time
    while timestep < nsteps:
      for r in reach_objects:
            #Need to get quc and qup
            upstream_flows = 0.0
            previous_upstream_flows = 0.0
            for id in r.upstream_ids: #Explicit loop reduces some overhead
              upstream_flows += flowveldepth[id, timestep, 0]
              previous_upstream_flows += flowveldepth[id, timestep-1, 0]
            #Index of segments required to process this reach
            segment_ids = [segment.id for segment in r.segments]

            if assume_short_ts:
                upstream_flows = previous_upstream_flows
            #Create compute reach kernel input buffer
            for i, segment in enumerate(r.segments):
              buf_view[i][0] = qlat_array[ segment.id, int(timestep/(nsteps/qlat_values.shape[1]))]
              buf_view[i][1] = segment.dt
              buf_view[i][2] = segment.dx
              buf_view[i][3] = segment.bw
              buf_view[i][4] = segment.tw
              buf_view[i][5] = segment.twcc
              buf_view[i][6] = segment.n
              buf_view[i][7] = segment.ncc
              buf_view[i][8] = segment.cs
              buf_view[i][9] = segment.s0
              buf_view[i][10] = flowveldepth[segment.id, timestep-1, 0]
              buf_view[i][11] = flowveldepth[segment.id, timestep-1, 1]
              buf_view[i][12] = flowveldepth[segment.id, timestep-1, 2]

            compute_reach_kernel(previous_upstream_flows, upstream_flows,
                                 len(r.segments), buf_view,
                                 out_buf,
                                 assume_short_ts)
            #Copy the output out
            for i, id in enumerate(segment_ids):
              flowveldepth[id, timestep, 0] = out_buf[i, 0]
              flowveldepth[id, timestep, 1] = out_buf[i, 1]
              flowveldepth[id, timestep, 2] = out_buf[i, 2]

      timestep += 1

    return np.asarray(data_idx, dtype=np.intp), np.asarray(flowveldepth.reshape(flowveldepth.shape[0], -1), dtype='float32')
