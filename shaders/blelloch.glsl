#version 460

#define THREAD_IDX              gl_LocalInvocationIndex
#define THREADS_NUM             64
#define THREAD_BLOCK_IDX        (gl_WorkGroupID.x + gl_NumWorkGroups.x * (gl_WorkGroupID.y + gl_NumWorkGroups.z * gl_WorkGroupID.z))
#define ITEMS_NUM               4u
#define BITSET_NUM              4u
#define BITSET_SIZE             16u

#define OP_UPSWEEP              0u  // 上向き走査 (リダクション)
#define OP_CLEAR_LAST           1u  // ルート要素のクリア
#define OP_DOWNSWEEP            2u  // 下向き走査 (分散)

layout(local_size_x = THREADS_NUM, local_size_y = 1, local_size_z = 1) in;

// バッファ定義
// [BITSET_SIZE * u_arr_len] サイズの配列
layout(std430, binding = 0) buffer ssbo_local_offsets_buf { uint b_local_offsets_buf[]; };

// ユニフォーム変数
uniform uint u_arr_len;     // 配列の論理的な長さ (2の冪乗が保証されている)
uniform uint u_depth;       // 現在のスキャンパスの深さ
uniform uint u_op;          // 実行する操作 (0, 1, 2)

// 関数定義

// バッファ内でのグローバルインデックスを計算
// 基数 (radix) ごとの連続した配列として扱う
uint to_partition_radixes_offsets_idx(uint radix, uint thread_block_idx)
{
    // radix * u_arr_len + thread_block_idx
    return radix * u_arr_len + thread_block_idx;
}

// ローカルメモリ内のインデックスを計算 (このコードでは未使用)
uint to_loc_idx(uint item_idx, uint thread_idx)
{
    return (thread_idx * ITEMS_NUM + item_idx);
}

// キー/ブロックインデックスを計算
// u_arr_len は THREAD_BLOCKS_NUM に対応する
uint to_key_idx(uint item_idx, uint thread_idx, uint thread_block_idx)
{
    // (ブロック開始インデックス) + (スレッド開始インデックス) + item_idx
    return (thread_block_idx * ITEMS_NUM * uint(THREADS_NUM)) + (thread_idx * ITEMS_NUM) + item_idx;
}

// main 関数

void main()
{
    // u_arr_len が2の冪乗であることをチェック (ログ関数を用いたチェック)
    if (uint(fract(log2(float(u_arr_len)))) != 0u) {
        // ERROR: The u_arr_len must be a power of 2 otherwise the Blelloch scan won't work!
        return; 
    }

    // ------------------------------------------------------------------------------------------------
    // Blelloch scan
    // ------------------------------------------------------------------------------------------------

    // 現在のステップサイズを計算 (2^depth)
    uint step = uint(exp2(float(u_depth)));

    if (u_op == OP_UPSWEEP)
    {
        // Reduce (upsweep): 隣接要素の合計を計算し、親ノードに格納
        for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
        {
            uint key_idx = to_key_idx(item_idx, THREAD_IDX, THREAD_BLOCK_IDX);
            
            // 現在のステップで処理を担当するインデックスかをチェック
            // key_idx が step * 2 の倍数である必要がある
            if (key_idx % (step * 2u) == 0u)
            {
                uint from_idx = key_idx + (step - 1u);
                uint to_idx = from_idx + step;

                if (to_idx < u_arr_len)
                {
                    // BITSET_SIZE (16) 個の基数すべてに対して累積和を計算
                    for (uint rad = 0u; rad < BITSET_SIZE; rad++)
                    {
                        uint from_rad_idx = to_partition_radixes_offsets_idx(rad, from_idx);
                        uint to_rad_idx = to_partition_radixes_offsets_idx(rad, to_idx);

                        // to_idxの位置に2つの要素の和を格納
                        b_local_offsets_buf[to_rad_idx] = b_local_offsets_buf[from_rad_idx] + b_local_offsets_buf[to_rad_idx];
                    }
                }
            }
        }
    }
    else if (u_op == OP_DOWNSWEEP)
    {
        // Downsweep: 累積和を分散させ、排他的プレフィックス和を計算
        for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
        {
            uint key_idx = to_key_idx(item_idx, THREAD_IDX, THREAD_BLOCK_IDX);
            
            // 現在のステップで処理を担当するインデックスかをチェック
            if (key_idx % (step * 2u) == 0u)
            {
                uint from_idx = key_idx + (step - 1u);
                uint to_idx = from_idx + step;

                if (to_idx < u_arr_len)
                {
                    // BITSET_SIZE (16) 個の基数すべてに対して累積和を計算
                    for (uint rad = 0u; rad < BITSET_SIZE; rad++)
                    {
                        uint from_rad_idx = to_partition_radixes_offsets_idx(rad, from_idx);
                        uint to_rad_idx = to_partition_radixes_offsets_idx(rad, to_idx);

                        // 1. to_idxの現在の累積和を保存
                        uint r = b_local_offsets_buf[to_rad_idx];
                        
                        // 2. to_idxを新しい累積和で更新
                        b_local_offsets_buf[to_rad_idx] = b_local_offsets_buf[from_rad_idx] + b_local_offsets_buf[to_rad_idx];
                        
                        // 3. from_idxを保存しておいた元のto_idxの値(r)で置き換え
                        //    これにより from_idx の位置には排他的プレフィックス和が格納される
                        b_local_offsets_buf[from_rad_idx] = r;
                    }
                }
            }
        }
    }
    else // u_op == OP_CLEAR_LAST
    {
        // Clear last: 配列の最後の要素をクリア (排他的スキャンの準備)
        for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
        {
            uint key_idx = to_key_idx(item_idx, THREAD_IDX, THREAD_BLOCK_IDX);
            
            // 配列の最後の要素を担当するスレッドのみが実行
            if (key_idx == (u_arr_len - 1u))
            {
                // BITSET_SIZE (16) 個の基数すべてに対して実行
                for (uint rad = 0u; rad < BITSET_SIZE; rad++)
                {
                    uint idx = to_partition_radixes_offsets_idx(rad, key_idx);
                    // ルートノード（最後の累積和）を0に設定
                    b_local_offsets_buf[idx] = 0u;
                }
            }
        }
    }
}