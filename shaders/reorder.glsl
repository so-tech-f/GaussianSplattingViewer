#version 460

#define THREAD_IDX              gl_LocalInvocationIndex
#define THREADS_NUM             64
#define THREAD_BLOCK_IDX        (gl_WorkGroupID.x + gl_NumWorkGroups.x * (gl_WorkGroupID.y + gl_NumWorkGroups.z * gl_WorkGroupID.z))
#define THREAD_BLOCKS_NUM       (gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_NumWorkGroups.z)
#define ITEMS_NUM               4u
#define BITSET_NUM              4u
#define BITSET_SIZE             16u

#define UINT32_MAX              uint(-1)

layout(local_size_x = THREADS_NUM, local_size_y = 1, local_size_z = 1) in;

// Read-only input keys
layout(std430, binding = 0) restrict readonly buffer in_keys_buf
{
    uint b_in_keys[];
};

// Write-only output keys
layout(std430, binding = 1) restrict writeonly buffer out_keys_buf
{
    uint b_out_keys[];
};

// Read-only input values (payload)
layout(std430, binding = 2) restrict readonly buffer in_values_buf
{
    uint b_in_values[];
};

// Write-only output values (payload)
layout(std430, binding = 3) restrict writeonly buffer out_values_buf
{
    uint b_out_values[];
};

// Block-level exclusive prefix sums (from previous Blelloch scan)
layout(std430, binding = 4) restrict readonly buffer local_offsets_buf
{
    uint b_local_offsets_buf[];
};

// Global counts for all radixes (total elements per radix)
layout(std430, binding = 5) restrict readonly buffer global_counts_buf
{
    uint b_glob_counts_buf[BITSET_SIZE];
};

// Uniform variables
uniform uint u_arr_len;         // Total length of the array
uniform uint u_bitset_idx;      // Current bitset (pass) index
uniform uint u_write_values;    // Flag to indicate if values should be written (0 or 1)

// Shared memory for local radix sort within the workgroup
shared uint s_prefix_sum[BITSET_SIZE][uint(THREADS_NUM) * ITEMS_NUM];
shared uint s_key_buf[uint(THREADS_NUM) * ITEMS_NUM][2];
shared uint s_sorted_indices[uint(THREADS_NUM) * ITEMS_NUM][2];
shared uint s_count[BITSET_SIZE]; // Unused in the final write phase

// Calculates the global index in the local_offsets_buf
uint to_partition_radixes_offsets_idx(uint radix, uint thread_block_idx)
{
    // Adjust thread blocks num to the next power of 2 for consistent indexing
    uint pow_of_2_thread_blocks_num = uint(exp2(ceil(log2(float(THREAD_BLOCKS_NUM)))));
    return radix * pow_of_2_thread_blocks_num + thread_block_idx;
}

// Calculates the local index within the shared memory partition (0 to THREADS_NUM * ITEMS_NUM - 1)
uint to_loc_idx(uint item_idx, uint thread_idx)
{
    return (thread_idx * ITEMS_NUM + item_idx);
}

// Calculates the global index in the input array for this item
uint to_key_idx(uint item_idx, uint thread_idx, uint thread_block_idx)
{
    return (thread_block_idx * ITEMS_NUM * uint(THREADS_NUM)) + (thread_idx * ITEMS_NUM) + item_idx;
}

void main()
{
    // Global offsets are calculated once per shader invocation.
    // This is an exclusive scan on the global counts to find the starting index for each radix group 
    // in the final output array.
    uint glob_off_buf[BITSET_SIZE];

    // Exclusive scan on global counts
    for (uint sum = 0u, i = 0u; i < BITSET_SIZE; i++)
    {
        glob_off_buf[i] = sum;
        sum += b_glob_counts_buf[i];
    }

    // ------------------------------------------------------------------------------------------------
    // 1. Load keys and values into shared memory
    // ------------------------------------------------------------------------------------------------

    for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
    {
        uint key_idx = to_key_idx(item_idx, THREAD_IDX, THREAD_BLOCK_IDX);
        uint loc_idx = to_loc_idx(item_idx, THREAD_IDX);

        // Load key. If out of bounds, use UINT32_MAX to place it at the end of the sorted array.
        s_key_buf[loc_idx][0] = key_idx < u_arr_len ? b_in_keys[key_idx] : UINT32_MAX;
        s_key_buf[loc_idx][1] = UINT32_MAX; // Double-buffering setup

        // Store the original local index to track payload (values)
        s_sorted_indices[loc_idx][0] = loc_idx;
        s_sorted_indices[loc_idx][1] = UINT32_MAX;
    }

    barrier();

    // ------------------------------------------------------------------------------------------------
    // 2. Local Radix Sort within the Partition
    // ------------------------------------------------------------------------------------------------

    // This loop performs a short radix sort on the elements within the workgroup (partition) 
    // up to the current bitset index (u_bitset_idx).
    uint in_partition_group_off[BITSET_SIZE];
    uint bitset_idx;

    // Loop up to and including the current sorting pass (u_bitset_idx)
    for (bitset_idx = 0u; bitset_idx <= u_bitset_idx; bitset_idx++)
    {
        uint bitset_mask = (BITSET_SIZE - 1u) << (BITSET_NUM * bitset_idx);

        // Init s_prefix_sum
        for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
        {
            for (uint bitset_val = 0u; bitset_val < BITSET_SIZE; bitset_val++)
            {
                uint loc_idx = to_loc_idx(item_idx, THREAD_IDX);
                s_prefix_sum[bitset_val][loc_idx] = 0u;
            }
        }
        barrier();

        // Predicate test: Count how many elements belong to each radix value.
        for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
        {
            uint loc_idx = to_loc_idx(item_idx, THREAD_IDX);
            uint k = s_key_buf[loc_idx][bitset_idx % 2u];
            uint radix = (k & bitset_mask) >> (BITSET_NUM * bitset_idx);
            s_prefix_sum[radix][loc_idx] = 1u; // Mark existence for the given radix
        }
        barrier();

        // Exclusive sum (Blelloch scan on shared memory):
        // This calculates the local offset for each element within its radix group.

        // Up-sweep
        uint log2_partition_size = uint(log2(float(uint(THREADS_NUM) * ITEMS_NUM)));
        for (uint d = 0u; d < log2_partition_size; d++)
        {
            for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
            {
                uint step = uint(exp2(float(d)));
                uint loc_idx = to_loc_idx(item_idx, THREAD_IDX);

                if (loc_idx % (step * 2u) == 0u)
                {
                    uint from_idx = loc_idx + (step - 1u);
                    uint to_idx = from_idx + step;

                    if (to_idx < uint(THREADS_NUM) * ITEMS_NUM)
                    {
                        for (uint bitset_val = 0u; bitset_val < BITSET_SIZE; bitset_val++)
                        {
                            s_prefix_sum[bitset_val][to_idx] = s_prefix_sum[bitset_val][from_idx] + s_prefix_sum[bitset_val][to_idx];
                        }
                    }
                }
            }
            barrier();
        }

        // Clear last (root)
        if (THREAD_IDX == 0u)
        {
            for (uint bitset_val = 0u; bitset_val < BITSET_SIZE; bitset_val++)
            {
                s_prefix_sum[bitset_val][(uint(THREADS_NUM) * ITEMS_NUM) - 1u] = 0u;
            }
        }
        barrier();

        // Down-sweep
        for (int d = int(log2_partition_size) - 1; d >= 0; d--)
        {
            for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
            {
                uint step = uint(exp2(float(d)));
                uint loc_idx = to_loc_idx(item_idx, THREAD_IDX);

                if (loc_idx % (step * 2u) == 0u)
                {
                    uint from_idx = loc_idx + (step - 1u);
                    uint to_idx = from_idx + step;

                    if (to_idx < uint(THREADS_NUM) * ITEMS_NUM)
                    {
                        for (uint bitset_val = 0u; bitset_val < BITSET_SIZE; bitset_val++)
                        {
                            uint r = s_prefix_sum[bitset_val][to_idx];
                            s_prefix_sum[bitset_val][to_idx] = r + s_prefix_sum[bitset_val][from_idx];
                            s_prefix_sum[bitset_val][from_idx] = r; // Exclusive swap
                        }
                    }
                }
            }
            barrier();
        }

        // Shuffling within shared memory (Local Reordering)
        
        // Determine the actual last valid index in the partition
        uint last_loc_idx;
        if (THREAD_BLOCK_IDX == (THREAD_BLOCKS_NUM - 1u)) {
            // Last block handling: calculate the index relative to the block start
            last_loc_idx = u_arr_len - (THREAD_BLOCKS_NUM - 1u) * (uint(THREADS_NUM) * ITEMS_NUM) - 1u;
        } else {
            // Full block
            last_loc_idx = (uint(THREADS_NUM) * ITEMS_NUM) - 1u;
        }

        // Calculate the starting offset for each radix group *within this partition*
        for (uint sum = 0u, i = 0u; i < BITSET_SIZE; i++)
        {
            in_partition_group_off[i] = sum;

            // Get the count of this radix group (from the last element of the exclusive scan + adjustment)
            bool is_last = ((s_key_buf[last_loc_idx][bitset_idx % 2u] & bitset_mask) >> (BITSET_NUM * bitset_idx)) == i;
            sum += s_prefix_sum[i][last_loc_idx] + (is_last ? 1u : 0u);
        }

        // Write the sorted elements to the next buffer (double buffering)
        for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
        {
            uint loc_idx = to_loc_idx(item_idx, THREAD_IDX);
            uint k = s_key_buf[loc_idx][bitset_idx % 2u];
            uint radix = (k & bitset_mask) >> (BITSET_NUM * bitset_idx);

            // Destination address within the shared memory partition (s_key_buf)
            // = Radix Group Start Offset + Element's Relative Index within the Radix Group
            uint dest_addr = in_partition_group_off[radix] + s_prefix_sum[radix][loc_idx];
            
            // Write to the output buffer
            s_key_buf[dest_addr][(bitset_idx + 1u) % 2u] = k; 
            s_sorted_indices[dest_addr][(bitset_idx + 1u) % 2u] = s_sorted_indices[loc_idx][bitset_idx % 2u];
        }

        barrier();
    } // End of Local Radix Sort loop (inner loop)

    // ------------------------------------------------------------------------------------------------
    // 3. Scattered writes to sorted partitions (Final Global Write)
    // ------------------------------------------------------------------------------------------------

    // bitset_idx is now u_bitset_idx + 1 from the loop exit, so we use u_bitset_idx directly for mask.
    uint final_bitset_mask = (BITSET_SIZE - 1u) << (BITSET_NUM * u_bitset_idx);

    // 最終的なローカル出力は (u_bitset_idx + 1) % 2u に格納されている
    uint final_buf_idx = (u_bitset_idx + 1u) % 2u;

    for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
        {
            uint key_idx = to_key_idx(item_idx, THREAD_IDX, THREAD_BLOCK_IDX);
        
            // Only process elements that are within the array length
            if (key_idx < u_arr_len)
            {
                uint loc_idx = to_loc_idx(item_idx, THREAD_IDX);
            
                // Read the key from the locally sorted buffer (output of the last inner pass)
                // Note: The correct buffer index is (u_bitset_idx + 1) % 2u 
                // after the loop finishes.
                uint k = s_key_buf[loc_idx][final_buf_idx];
                uint rad = (k & final_bitset_mask) >> (BITSET_NUM * u_bitset_idx);

                // 1. Global Offset: Start of the radix group in the entire array
                uint glob_off = glob_off_buf[rad];
            
                // 2. Local Block Offset: Start of this block's radix group within the global radix group
                uint local_off = b_local_offsets_buf[to_partition_radixes_offsets_idx(rad, THREAD_BLOCK_IDX)];

                // 3. Index within Radix Group: Element's position relative to the start of its radix group *in this block*
                // The local sort ensures 'loc_idx' is the sorted position, and 'in_partition_group_off[rad]' 
                // is the start of that radix group in the sorted local buffer.
                uint index_in_group = loc_idx - in_partition_group_off[rad];

                // Final Destination Index (Global Position)
                uint dest_idx = glob_off + local_off + index_in_group;

                // Write key to the final position
                b_out_keys[dest_idx] = k;
            
                // Write value (payload) using the tracked original index
                if (u_write_values != 0u)
                {
                    // Calculate the original global index: Block Start + Original Local Index
                    uint original_glob_idx = THREAD_BLOCK_IDX * (uint(THREADS_NUM) * ITEMS_NUM) + s_sorted_indices[loc_idx][final_buf_idx];
                    b_out_values[dest_idx] = b_in_values[original_glob_idx];
                }
            }
        }
}