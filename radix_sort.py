import os
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
os.environ["GALLIUM_DRIVER"] = "d3d12"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "0"

import numpy as np
import ctypes
from OpenGL.GL import *
import glfw

# --- 基本的な OpenGL ラッパークラス ---

class Shader:
    """OpenGLシェーダーオブジェクトを管理するクラス"""
    def __init__(self, shader_type):
        self.name = glCreateShader(shader_type)
    
    def __del__(self):
        # オブジェクトが破棄されるときにシェーダーを削除
        glDeleteShader(self.name)
    
    def load_source(self, src):
        # シェーダーソースコードを設定
        glShaderSource(self.name, src)
    
    def compile(self):
        # シェーダーをコンパイル
        glCompileShader(self.name)
        status = glGetShaderiv(self.name, GL_COMPILE_STATUS)
        if not status:
            log = glGetShaderInfoLog(self.name)
            # UnicodeDecodeErrorを避けるためデコード
            raise RuntimeError(f"Shader compilation failed: {log.decode()}")

class Program:
    """OpenGLプログラムオブジェクトを管理するクラス"""
    def __init__(self):
        self.name = glCreateProgram()
    
    def __del__(self):
        # オブジェクトが破棄されるときにプログラムを削除
        glDeleteProgram(self.name)
    
    def attach_shader(self, shader_name):
        # シェーダーをプログラムにアタッチ
        glAttachShader(self.name, shader_name)
    
    def link(self):
        # プログラムをリンク
        glLinkProgram(self.name)
        status = glGetProgramiv(self.name, GL_LINK_STATUS)
        if not status:
            log = glGetProgramInfoLog(self.name)
            raise RuntimeError(f"Program linking failed: {log.decode()}")
    
    def get_uniform_location(self, name):
        # ユニフォーム変数のロケーションを取得
        loc = glGetUniformLocation(self.name, name)
        # GLSLの最適化で使われない変数がある可能性も考慮し、
        # 厳密なエラーチェックは環境による
        if loc < 0:
             # Uniformが見つからない場合は警告を出す方が実用的
             pass
        return loc
    
    def use(self):
        # このプログラムを使用
        glUseProgram(self.name)

# --- OpenGLコンテキスト管理関数 ---

def init_opengl_context():
    """OpenGLコンテキストを初期化し、オフスクリーンウィンドウを作成"""
    if not glfw.init():
        raise RuntimeError("GLFWの初期化に失敗")
    
    # ウィンドウを非表示に設定（オフスクリーン処理用）
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(1, 1, "Hidden", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("ウィンドウの作成に失敗")
    
    glfw.make_context_current(window)
    return window

def read_buffer_data(buffer_obj, count, dtype=np.uint32):
    """SSBOからデータを読み取るヘルパー関数"""
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_obj)
    
    # NumPy配列を作成
    result = np.empty(count, dtype=dtype)
    
    # ctypesポインタを取得
    ptr = result.ctypes.data_as(ctypes.c_void_p)
    
    # データをコピー
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, result.nbytes, ptr)
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return result

from math import ceil, log2, exp2

# --- 定数定義 ---
THREADS_PER_BLOCK = 64
ITEMS_PER_THREAD = 4
BITSET_NUM = 4              # 1パスあたり処理するビット数
BITSET_SIZE = 16            # 2^BITSET_NUM = 16
BITSET_COUNT = 8            # 32 bits / 4 bits per pass = 8 passes

class RadixSorter:
    def __init__(self, init_arr_len):
        # シェーダーソースの読み込み（ファイルパスは環境依存）
        self.count_shader_src = open("shaders/count.glsl", "r").read()
        self.offset_shader_src = open("shaders/blelloch.glsl", "r").read()
        self.reorder_shader_src = open("shaders/reorder.glsl", "r").read()
        
        # プログラムの初期化
        self._init_count_program()
        self._init_offset_program()
        self._init_reorder_program()
        
        # バッファの初期化とリサイズ
        self.local_offsets_buf = 0
        self.glob_counts_buf = 0
        self.keys_scratch_buf = 0
        self.values_scratch_buf = 0
        self.internal_arr_len = 0
        self._resize_internal_buffers(init_arr_len)
    
    # --- プライベートメソッド: プログラム初期化 ---

    def _init_count_program(self):
        shader = Shader(GL_COMPUTE_SHADER)
        shader.load_source(self.count_shader_src)
        shader.compile()
        
        self.count_program = Program()
        self.count_program.attach_shader(shader.name)
        self.count_program.link()
    
    def _init_offset_program(self):
        shader = Shader(GL_COMPUTE_SHADER)
        shader.load_source(self.offset_shader_src)
        shader.compile()
        
        self.offset_program = Program()
        self.offset_program.attach_shader(shader.name)
        self.offset_program.link()
    
    def _init_reorder_program(self):
        shader = Shader(GL_COMPUTE_SHADER)
        shader.load_source(self.reorder_shader_src)
        shader.compile()
        
        self.reorder_program = Program()
        self.reorder_program.attach_shader(shader.name)
        self.reorder_program.link()
    
    # --- プライベートメソッド: サイズ計算 ---

    def _calc_thread_blocks_num(self, arr_len):
        """必要なスレッドブロック数を計算"""
        elements_per_block = THREADS_PER_BLOCK * ITEMS_PER_THREAD
        return int(ceil(arr_len / elements_per_block))
    
    def _round_to_power_of_2(self, dim):
        """2の冪乗に切り上げ（Blellochスキャン用）"""
        # dimが0の場合の例外処理
        if dim <= 0: return 1
        return int(exp2(ceil(log2(dim))))
    
    # --- プライベートメソッド: バッファ管理 ---

    def _resize_internal_buffers(self, arr_len):
        """内部ストレージバッファのサイズ変更（または初回生成）"""
        self.internal_arr_len = arr_len
        
        thread_blocks = self._calc_thread_blocks_num(arr_len)
        pow2_blocks = self._round_to_power_of_2(thread_blocks)
        
        # 既存バッファがあれば削除
        if self.local_offsets_buf != 0:
             glDeleteBuffers(4, [self.local_offsets_buf, self.glob_counts_buf, self.keys_scratch_buf, self.values_scratch_buf])

        # 1. local_offsets_buf: 各ブロックの基数ごとのローカルカウントを累積和するためのバッファ
        self.local_offsets_buf = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.local_offsets_buf)
        # size: 2の冪乗ブロック数 * 基数サイズ * sizeof(uint32)
        glBufferData(GL_SHADER_STORAGE_BUFFER, pow2_blocks * BITSET_SIZE * 4, None, GL_DYNAMIC_DRAW)
        
        # 2. glob_counts_buf: 全配列の基数ごとのグローバルカウント（ヒストグラム）
        self.glob_counts_buf = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.glob_counts_buf)
        glBufferData(GL_SHADER_STORAGE_BUFFER, BITSET_SIZE * 4, None, GL_DYNAMIC_DRAW)
        
        # 3. keys_scratch_buf, 4. values_scratch_buf: ダブルバッファリング用のスクラッチ領域
        self.keys_scratch_buf = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.keys_scratch_buf)
        glBufferData(GL_SHADER_STORAGE_BUFFER, arr_len * 4, None, GL_STATIC_DRAW)
        
        self.values_scratch_buf = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.values_scratch_buf)
        glBufferData(GL_SHADER_STORAGE_BUFFER, arr_len * 4, None, GL_STATIC_DRAW)
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # --- プライベートメソッド: GPUパス実行 ---

    def _count_pass(self, key_buf, arr_len, bitset_idx, thread_blocks_num):
        """1. カウントフェーズ (ヒストグラム生成)"""
        self.count_program.use()
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, key_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.local_offsets_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.glob_counts_buf)
        
        glUniform1ui(self.count_program.get_uniform_location("u_arr_len"), arr_len)
        glUniform1ui(self.count_program.get_uniform_location("u_bitset_idx"), bitset_idx)
        
        glDispatchCompute(thread_blocks_num, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _blelloch_scan(self, pow2_blocks):
        """2. オフセット計算フェーズ (Blellochスキャン)"""
        self.offset_program.use()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.local_offsets_buf)
        
        # Up-sweep (リダクション)
        log2_blocks = int(log2(pow2_blocks))
        for d in range(log2_blocks):
            self._dispatch_offset_op(pow2_blocks, 0, d) # OP_UPSWEEP
        
        # Clear last (ルートノードを0に)
        self._dispatch_offset_op(pow2_blocks, 1, 0) # OP_CLEAR_LAST
        
        # Down-sweep (分散)
        for d in range(log2_blocks - 1, -1, -1):
            self._dispatch_offset_op(pow2_blocks, 2, d) # OP_DOWNSWEEP

    def _dispatch_offset_op(self, pow2_blocks, op, depth):
        """Blellochスキャン用のディスパッチヘルパー"""
        glUniform1ui(self.offset_program.get_uniform_location("u_arr_len"), pow2_blocks)
        glUniform1ui(self.offset_program.get_uniform_location("u_op"), op)
        glUniform1ui(self.offset_program.get_uniform_location("u_depth"), depth)
        
        elements_per_block = THREADS_PER_BLOCK * ITEMS_PER_THREAD
        workgroups = int(ceil(pow2_blocks / elements_per_block))
        
        glDispatchCompute(workgroups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _reorder_pass(self, keys_buffers, values_buffers, arr_len, 
                      pass_idx, thread_blocks_num, write_values):
        """3. リオーダーフェーズ (要素の再配置)"""
        self.reorder_program.use()
        
        read_idx = pass_idx % 2
        write_idx = (pass_idx + 1) % 2

        # 0: in_keys_buf, 1: out_keys_buf
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, keys_buffers[read_idx], 0, arr_len * 4)
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, keys_buffers[write_idx], 0, arr_len * 4)
        
        if write_values:
            # 2: in_values_buf (current read buffer), 3: out_values_buf (current write buffer)
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, values_buffers[read_idx], 0, arr_len * 4)
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, values_buffers[write_idx], 0, arr_len * 4)
        
        # 4: local_offsets_buf, 5: global_counts_buf
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.local_offsets_buf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.glob_counts_buf)
        
        glUniform1ui(self.reorder_program.get_uniform_location("u_write_values"), 1 if write_values else 0)
        glUniform1ui(self.reorder_program.get_uniform_location("u_arr_len"), arr_len)
        glUniform1ui(self.reorder_program.get_uniform_location("u_bitset_idx"), pass_idx)
        
        glDispatchCompute(thread_blocks_num, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    # --- パブリックメソッド: ソート実行 ---

    def sort(self, key_buf, val_buf, arr_len):
        """
        キーと値のGPUバッファを受け取り、並列基数ソートを実行する。
        結果は元のバッファに書き戻されるか、スクラッチバッファに残る。
        """
        if arr_len <= 1:
            return
        
        if self.internal_arr_len < arr_len:
            self._resize_internal_buffers(arr_len)
        
        thread_blocks_num = self._calc_thread_blocks_num(arr_len)
        pow2_blocks = self._round_to_power_of_2(thread_blocks_num)
        
        keys_buffers = [key_buf, self.keys_scratch_buf]
        values_buffers = [val_buf, self.values_scratch_buf]
        
        # 8パスの基数ソート（32ビット整数を4ビットずつ8回処理）
        for pass_idx in range(BITSET_COUNT):
            # 内部バッファのクリア
            zero = np.array([0], dtype=np.uint32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.glob_counts_buf)
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, zero)
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.local_offsets_buf)
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, zero)
            
            # 1. カウント (ヒストグラム生成)
            self._count_pass(keys_buffers[pass_idx % 2], arr_len, pass_idx, thread_blocks_num)
            
            # 2. Blellochスキャン (オフセット計算)
            self._blelloch_scan(pow2_blocks)
            
            # 3. リオーダー (要素の再配置)
            write_values = (val_buf is not None and val_buf != 0)
            self._reorder_pass(keys_buffers, values_buffers, arr_len, pass_idx, thread_blocks_num, write_values)
        
        glUseProgram(0)

# --- 実行コード (テスト用エントリポイント) ---

def main():
    # 1. OpenGLコンテキストの初期化
    window = init_opengl_context()
    
    # 2. ソーターの作成
    sorter = RadixSorter(100000)
    
    # 3. テストデータの生成
    ARR_LEN = 100000
    values = np.random.randint(0, 10000000, ARR_LEN, dtype=np.uint32)
    np.savetxt('values.csv', values, fmt='%u', delimiter=',')
    indices = np.arange(ARR_LEN, dtype=np.uint32)

    # 4. GPUバッファの作成とデータ転送
    values_buf = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, values_buf)
    glBufferData(GL_SHADER_STORAGE_BUFFER, values.nbytes, values, GL_DYNAMIC_DRAW)

    indices_buf = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indices_buf)
    glBufferData(GL_SHADER_STORAGE_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # 5. ソート実行
    print(f"ソート対象の要素数: {ARR_LEN}")
    import time
    start_time = time.time()
    sorter.sort(values_buf, indices_buf, len(values))
    end_time = time.time()
    print(f"ソートにかかった時間: {end_time - start_time:.6f} 秒")

    # 6. 結果の読み取り
    sorted_values = read_buffer_data(values_buf, len(values))
    sorted_indices = read_buffer_data(indices_buf, len(indices))

    # 7. 厳密な検証と表示
    print("\n--- 結果の検証 (詳細) ---")
    print(f"ソート前 (Values) sample: {values[:10]}")
    print(f"ソート後 (Values) sample: {sorted_values[:10]}")
    print(f"対応するインデックス (Indices) sample: {sorted_indices[:10]}")

    # CPUによる期待値 (値の昇順) とソート済みインデックス
    cpu_sorted_values = np.sort(values)
    # Use a stable sort so equal keys preserve original relative order
    cpu_argsort = np.argsort(values, kind='stable')

    # 比較: 値
    values_match = np.array_equal(sorted_values, cpu_sorted_values)

    # 比較: インデックスが元のインデックスに一致するか (payload の追跡が正しいか)
    expected_indices = indices[cpu_argsort]
    indices_match = np.array_equal(sorted_indices, expected_indices)

    print(f"インデックス一致: {'OK' if indices_match else 'NG'}")
    print(f"値一致: {'OK' if values_match else 'NG'}")

    if not values_match:
        # 差分の最初の箇所を出力
        diffs = np.nonzero(sorted_values != cpu_sorted_values)[0]
        first = diffs[0] if diffs.size > 0 else None
        print(f"値差分の最初のインデックス: {first}")
        if first is not None:
            print(f" GPU: {sorted_values[first-2:first+3]}")
            print(f" CPU: {cpu_sorted_values[first-2:first+3]}")

    if not indices_match:
        diffs = np.nonzero(sorted_indices != expected_indices)[0]
        first = diffs[0] if diffs.size > 0 else None
        print(f"インデックス差分の最初のインデックス: {first}")
        if first is not None:
            print(f" GPU vals: {sorted_indices[first-5:first+5]}")
            print(f" Exp vals: {expected_indices[first-5:first+5]}")

    # CSV に出力（デバッグ用）: 全配列と不一致箇所の差分を保存
    try:
        # 全体のGPU値 / 期待値をそれぞれ保存
        np.savetxt('gpu_values.csv', sorted_values, fmt='%u', delimiter=',')
        np.savetxt('exp_values.csv', cpu_sorted_values, fmt='%u', delimiter=',')
    except Exception as e:
        print(f"CSV 保存に失敗しました: {e}")

    print(f"\nソート結果の確認: {'成功' if (indices_match and values_match) else '失敗'}")

    # 8. クリーンアップ
    glDeleteBuffers(2, [values_buf, indices_buf])
    glfw.terminate()


if __name__ == '__main__':
    main()