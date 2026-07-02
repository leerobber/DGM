//! Graph BFS WASM agent.
//! Receives a KernelBlock with intent=41 (QUERY_GRAPH/BFS).
//! Transforms the first word to RESULT type and echoes payload_ref (start_id).
#![no_std]
use core::slice;

// ── output buffer ─────────────────────────────────────────────────────────────
// Layout: [len:u32 LE][agent_id:u32][genome_id:u32][creds_token:u32][task_id:u32]
//         [n_words:u32=1][word:u64][metrics_ref:u32]  →  4+32 = 36 bytes
static mut OUT_BUF: [u8; 64] = [0u8; 64];

// ── constants ─────────────────────────────────────────────────────────────────
const TYPE_RESULT: u8 = 6;
const INTENT_OUT: u8 = 41; // QUERY_GRAPH — BFS result

// ── helpers ───────────────────────────────────────────────────────────────────

#[inline(always)]
fn r_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]])
}

#[inline(always)]
fn r_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([b[o],b[o+1],b[o+2],b[o+3],b[o+4],b[o+5],b[o+6],b[o+7]])
}

#[inline(always)]
fn w_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o+4].copy_from_slice(&v.to_le_bytes());
}

#[inline(always)]
fn w_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o+8].copy_from_slice(&v.to_le_bytes());
}

// SemanticWord: type[63:56]|intent[55:48]|channel[47:40]|priority[39:32]|confidence[31:16]|payload_ref[15:0]
#[inline(always)]
fn sw_decode(w: u64) -> (u8, u8, u8, u8, u16, u16) {
    (((w>>56)&0xFF) as u8, ((w>>48)&0xFF) as u8, ((w>>40)&0xFF) as u8,
     ((w>>32)&0xFF) as u8, ((w>>16)&0xFFFF) as u16, (w&0xFFFF) as u16)
}

#[inline(always)]
fn sw_encode(ty: u8, intent: u8, ch: u8, pri: u8, conf: u16, pref: u16) -> u64 {
    ((ty as u64)<<56)|((intent as u64)<<48)|((ch as u64)<<40)
    |((pri as u64)<<32)|((conf as u64)<<16)|(pref as u64)
}

// ── WASM entry ────────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn handle_block(ptr: *const u8, len: usize) -> *mut u8 {
    let input = slice::from_raw_parts(ptr, len);
    let buf = &mut OUT_BUF;

    if len < 24 {
        w_u32(buf, 0, 0);
        return buf.as_mut_ptr();
    }

    let agent_id    = r_u32(input, 0);
    let genome_id   = r_u32(input, 4);
    let creds_token = r_u32(input, 8);
    let task_id     = r_u32(input, 12);
    let n_words     = r_u32(input, 16) as usize;
    let word        = if n_words > 0 && len >= 28 { r_u64(input, 20) } else { 0 };
    let words_end   = 20 + n_words * 8;
    let metrics_ref = if len >= words_end + 4 { r_u32(input, words_end) } else { 0 };

    let (_, _, ch, pri, conf, pref) = sw_decode(word);
    // Boost confidence; echo payload_ref (start node id for BFS)
    let result_word = sw_encode(TYPE_RESULT, INTENT_OUT, ch, pri,
                                conf.saturating_add(500).min(0xFFFF), pref);

    // Write output: [len:u32][KernelBlock body (32 bytes)]
    w_u32(buf,  0, 32);
    w_u32(buf,  4, agent_id);
    w_u32(buf,  8, genome_id);
    w_u32(buf, 12, creds_token);
    w_u32(buf, 16, task_id);
    w_u32(buf, 20, 1);            // n_words = 1
    w_u64(buf, 24, result_word);
    w_u32(buf, 32, metrics_ref);

    buf.as_mut_ptr()
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
