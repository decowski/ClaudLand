/*
 * Fast decoder for KamLAND ATWD waveform banks.
 *
 * Compiled on first import by claudland/fastdecode.py (``cc -O2 -shared``) and
 * called through ctypes.  Two entry points:
 *
 *   kl_decode_cmp   -- CmpATWD / CmpAntiATWD banks (Huffman "DiffDetEntStat"
 *                      compression, plus embedded raw Thorsten blocks)
 *   kl_decode_atwd  -- uncompressed ATWD / AntiATWD banks (43 x u32 per waveform)
 *
 * Both walk the HitHeader words (cable = w & 0xfff, n_waveforms = (w>>12)&7)
 * and fill meta[n][5] = {cable, atwd, gain, launch_offset, flags} and
 * samples[n][128] (raw ADC counts, pedestal restored, 0..1023).  flags:
 * bit 0 = Huffman bit count outside the padded word, bit 1 = raw Thorsten block.
 * The algorithms are bit-for-bit ports of HuffmanWaveformDecoder in wfcomp.py.
 */
#include <stdint.h>

#define NS 128
#define DSTART 18
#define THORSTEN_BITS 1376
#define N_GAINS 4

/* little-endian u32 word k of a payload of plen bytes; missing bytes read as 0 */
static inline uint64_t rd_word(const uint8_t *p, long plen, long k)
{
    uint64_t w = 0;
    long o = 4 * k;
    for (int i = 0; i < 4; i++)
        if (o + i < plen)
            w |= (uint64_t)p[o + i] << (8 * i);
    return w;
}

/* n (<= 32) bits, MSB first, starting at bit position pos */
static inline int peek(const uint8_t *p, long plen, long pos, int n)
{
    long k = pos >> 5;
    int o = (int)(pos & 31);
    uint64_t comb = (rd_word(p, plen, k) << 32) | rd_word(p, plen, k + 1);
    return (int)((comb >> (64 - o - n)) & ((1u << n) - 1));
}

static inline int decode_launch(int b)
{
    return (b & 0x80) ? -(b & 0x7F) : b;
}

/* Huffman decode + inverse transform + pedestal restore of one block. Returns bits used (or -1). */
static long decode_block(const uint8_t *p, long plen, int nbits,
                         const int32_t *table, const int32_t *tree, const int32_t *nb,
                         const uint16_t *ped, int16_t *out)
{
    int d[NS];
    long pos = 0;
    long nwords = (nbits + 31) >> 5;
    if (plen > 4 * nwords)
        plen = 4 * nwords;              /* the Python decoder truncates to nwords words */
    for (int i = 0; i < NS; i++) {
        int sym;
        int t = peek(p, plen, pos, 12);
        int v = table[t];
        if (v >= 2048) {
            sym = v - 2048;
            pos += nb[sym];
        } else {
            int ptr = v;
            pos += 12;
            for (;;) {
                int bit = peek(p, plen, pos, 1);
                pos += 1;
                if (ptr + bit < 0 || ptr + bit >= 2048)
                    return -1;
                int node = tree[ptr + bit];
                if (node >= 2048) { sym = node - 2048; break; }
                ptr = node;
                if (pos > 32 * nwords + 64)
                    return -1;          /* runaway: corrupt block */
            }
        }
        d[i] = sym;
    }
    /* undo power-compression shift */
    int dsav = 0;
    for (int m = NS - 1; m >= 0; m--) {
        if (((dsav + 2) & 1023) > 4) {
            int shift = (dsav & 512) ? dsav + 2 : dsav - 2;
            d[m] = (d[m] + shift) & 1023;
        }
        dsav = d[m];
    }
    /* undo reflection */
    int dold = 0;
    for (int m = NS - 1; m >= 0; m--) {
        if ((dold & 512) == 0)
            d[m] = (-d[m]) & 1023;
        dold = d[m];
    }
    /* difference decode, add pedestal, mask to 10 bits */
    long acc = DSTART;
    for (int m = NS - 1; m >= 0; m--) {
        acc += d[m];
        out[m] = (int16_t)((acc + ped[m]) & 1023);
    }
    return pos;
}

/* 43 little-endian u32 words: word w holds samples 3*(42-w)+k in bits 10*k */
static void decode_thorsten(const uint8_t *p, long plen, int16_t *out)
{
    for (int w = 0; w < 43; w++) {
        uint64_t val = rd_word(p, plen, w);
        int base = 3 * (42 - w);
        for (int k = 0; k < 3; k++) {
            int s = base + k;
            if (s < NS)
                out[s] = (int16_t)((val >> (10 * k)) & 0x3FF);
        }
    }
}

/*
 * Returns the number of decoded waveforms, or a negative error code:
 *  -1 bank truncated, -2 output arrays too small, -3 corrupt Huffman block,
 *  -4 cable outside the connection table.
 */
long kl_decode_cmp(const uint8_t *cmp, long cmp_len,
                   const uint16_t *words, long nwords,
                   const uint8_t *want,            /* [4] flags per gain */
                   const int64_t *base_row, long n_cables,
                   const uint16_t *ped, long n_ped_rows,
                   const int32_t *table, const int32_t *tree, const int32_t *nb,
                   int32_t *meta, int16_t *samples, long max_n)
{
    static const uint16_t zero_ped[NS] = {0};
    long pos = 0, n = 0;
    for (long h = 0; h < nwords; h++) {
        int cable = words[h] & 0xFFF;
        int nwf = (words[h] >> 12) & 7;
        for (int j = 0; j < nwf; j++) {
            if (pos + 6 > cmp_len)
                return -1;
            long size = cmp[pos];
            int launch = decode_launch(cmp[pos + 1]);
            int flags = cmp[pos + 2];
            int nbits = cmp[pos + 4] | (cmp[pos + 5] << 8);
            int gain = flags & 3;
            int atwd = (flags >> 2) & 1;
            if (pos + size > cmp_len || size < 6)
                return -1;
            if (want[gain]) {
                if (n >= max_n)
                    return -2;
                int16_t *out = samples + n * NS;
                int mism = 0;
                if (nbits == THORSTEN_BITS) {
                    decode_thorsten(cmp + pos + 6, size - 6, out);
                    mism = 2;
                } else {
                    const uint16_t *pd = zero_ped;
                    if (cable < n_cables && base_row[cable] >= 0) {
                        long row = base_row[cable] + atwd * N_GAINS + gain;
                        if (row < n_ped_rows)
                            pd = ped + row * NS;
                    }
                    long used = decode_block(cmp + pos + 6, size - 6, nbits, table, tree, nb, pd, out);
                    if (used < 0)
                        return -3;
                    mism = !(used > nbits - 32 && used <= nbits);  /* nbits is padded to 32-bit words */
                }
                int32_t *m = meta + n * 5;
                m[0] = cable; m[1] = atwd; m[2] = gain; m[3] = launch; m[4] = mism;
                n++;
            }
            pos += size;
        }
    }
    return n;
}

/* Uncompressed ATWD bank: 43 u32 words per waveform (SFATWDBank.cc layout). */
long kl_decode_atwd(const uint32_t *data, long ndata,
                    const uint16_t *words, long nwords,
                    const uint8_t *want,
                    int32_t *meta, int16_t *samples, long max_n)
{
    long idx = 0, n = 0;
    for (long h = 0; h < nwords; h++) {
        int cable = words[h] & 0xFFF;
        int nwf = (words[h] >> 12) & 7;
        for (int j = 0; j < nwf; j++) {
            if (idx + 43 > ndata)
                return -1;
            const uint32_t *blk = data + idx;
            idx += 43;
            uint32_t head = blk[0];
            int atwd = (head >> 30) & 1;
            int gain = (head >> 28) & 3;
            if (!want[gain])
                continue;
            if (n >= max_n)
                return -2;
            int16_t *s = samples + n * NS;
            s[127] = (int16_t)((head >> 10) & 0x3FF);
            s[126] = (int16_t)(head & 0x3FF);
            for (int k = 0; k < 42; k++) {
                uint32_t body = blk[1 + k];
                s[125 - 3 * k] = (int16_t)((body >> 20) & 0x3FF);
                s[124 - 3 * k] = (int16_t)((body >> 10) & 0x3FF);
                s[123 - 3 * k] = (int16_t)(body & 0x3FF);
            }
            int32_t *m = meta + n * 5;
            m[0] = cable; m[1] = atwd; m[2] = gain; m[3] = decode_launch((head >> 20) & 0xFF); m[4] = 0;
            n++;
        }
    }
    return n;
}
