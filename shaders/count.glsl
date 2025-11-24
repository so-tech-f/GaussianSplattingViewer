#version 460

#define THREAD_IDX              gl_LocalInvocationIndex
#define THREADS_NUM             64
#define THREAD_BLOCK_IDX        (gl_WorkGroupID.x + gl_NumWorkGroups.x * (gl_WorkGroupID.y + gl_NumWorkGroups.z * gl_WorkGroupID.z))
#define THREAD_BLOCKS_NUM       (gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_NumWorkGroups.z)
#define ITEMS_NUM               4u

#define BITSET_NUM              4u
#define BITSET_SIZE             16u // 2^BITSET_NUM

layout(local_size_x = THREADS_NUM, local_size_y = 1, local_size_z = 1) in;

// バッファ定義
layout(std430, binding = 0) buffer ssbo_key              { uint b_key_buf[]; };                  // 入力キー配列
layout(std430, binding = 1) buffer ssbo_count_buf        { uint b_count_buf[]; };                // ブロックごとの基数カウント [BITSET_SIZE * 2^ceil(log2(THREAD_BLOCKS_NUM))]
layout(std430, binding = 2) buffer ssbo_tot_count_buf    { uint b_tot_count_buf[BITSET_SIZE]; }; // 全体の基数カウント

// ユニフォーム変数
uniform uint u_arr_len;     // キー配列の長さ
uniform uint u_bitset_idx;  // 処理するビットセットのインデックス (例: 0, 1, 2, ...)

// 関数定義

// ブロックごとの基数カウントバッファにおけるインデックスを計算
// padding to power of 2: pow_of_2_thread_blocks_num
uint to_partition_radixes_offsets_idx(uint radix, uint thread_block_idx)
{
    // THREAD_BLOCKS_NUMを2の冪乗に切り上げてパディングサイズを計算
    uint pow_of_2_thread_blocks_num = uint(exp2(ceil(log2(float(THREAD_BLOCKS_NUM)))));
    // (radix * padding_size) + thread_block_idx
    return radix * pow_of_2_thread_blocks_num + thread_block_idx;
}

// ローカルメモリ内のインデックスを計算 (このコードでは未使用)
uint to_loc_idx(uint item_idx, uint thread_idx)
{
    return (thread_idx * ITEMS_NUM + item_idx);
}

// グローバルキー配列 (b_key_buf) におけるインデックスを計算
uint to_key_idx(uint item_idx, uint thread_idx, uint thread_block_idx)
{
    // (ブロック開始インデックス) + (スレッド開始インデックス) + item_idx
    return (thread_block_idx * ITEMS_NUM * uint(THREADS_NUM)) + (thread_idx * ITEMS_NUM) + item_idx;
}

// main 関数

void main()
{
    // 各スレッドはITEMS_NUM個のキーを処理
    for (uint item_idx = 0u; item_idx < ITEMS_NUM; item_idx++)
    {
        uint key_idx = to_key_idx(item_idx, THREAD_IDX, THREAD_BLOCK_IDX);
        
        // 配列の境界チェック
        if (key_idx >= u_arr_len) {
            continue;
        }

        // 処理中のビットセットを抽出するためのマスクとシフトを計算
        // BITSET_NUM = 4 => 4ビットを処理
        // BITSET_SIZE = 16 => 0x0F (4ビットマスク)
        // bitset_mask = (16 - 1) << (4 * u_bitset_idx)
        uint bitset_mask = (BITSET_SIZE - 1u) << (BITSET_NUM * u_bitset_idx);
        
        // キーから現在のビットセットに対応する基数を抽出 (0から15の範囲)
        // rad = (b_key_buf[key_idx] & bitset_mask) >> (BITSET_NUM * u_bitset_idx);
        uint rad = (b_key_buf[key_idx] & bitset_mask) >> (BITSET_NUM * u_bitset_idx);

        // スレッドブロックごとの基数カウントをインクリメント (グローバルアトミック操作)
        atomicAdd(b_count_buf[to_partition_radixes_offsets_idx(rad, THREAD_BLOCK_IDX)], 1u);
        
        // 全体の基数カウントをインクリメント (グローバルアトミック操作)
        atomicAdd(b_tot_count_buf[rad], 1u);
    }
}