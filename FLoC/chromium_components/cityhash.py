# taken from [1]
# [1] https://github.com/prashnts/Hashes/blob/master/Hashes/cityhash.py

"""
Python implementation of Google's CityHash.
https://code.google.com/p/cityhash/
The process is being reproduced as it was in original implementation.
Additional source comments are added as and when required.
"""

##Copyright (c) 2011 Google, Inc.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 #CityHash, by Geoff Pike and Jyrki Alakuijala

# Some primes between 2^63 and 2^64 for various usage.
K0 = 0xc3a5c85c97cb3127 #14097894508562428199
K1 = 0xb492b66fbe98f273 #13011662864482103923
K2 = 0x9ae16a3b2f90404f #11160318154034397263
K3 = 0xc949d7c7509e6557 #14504361325974414679

def lower32(candidate):
    """
    Returns last 32 bits from candidate.
    """
    return candidate & 0xffffffff

def lower64(candidate):
    """
    Returns last 64 bits from candidate.
    """
    return candidate & 0xffffffffffffffff

def higher64(candidate):
    """
    Returns higher than 64 bits from candidate, by shifting lower end to right.
    """
    return candidate >> 64

def bytes(candidate):
    """
    Returns the hex-equivalent of byte-string.
    """
    result = 0x0

    # Reversed the candidate, and builds a list of characters returned as their corresponding Unicode code point.
    # https://docs.python.org/2/library/functions.html#ord
    characters = list(ord(character) for character in candidate[::-1])

    for character in characters:
        # Left shift 8 bits, and then add the current character. Automatically
        # assumes the ASCII input of string.
        result <<= 8
        result |= character

    return result

def hash128to64(candidate):
    """
    Hashes 128 input bits down to 64 bits of output.
    """
    kMul = 0x9ddfea08eb382d69 #11376068507788127593
    a = lower64((lower64(candidate) ^ higher64(candidate)) * kMul)
    a ^= (a >> 47)
    b = lower64((higher64(candidate) ^ a) * kMul)
    b ^= (b >> 47)
    b = b * kMul
    return lower64(b)

def rotate(val, shift):
    """
    Bitwise right rotate.
    """
    return val if shift == 0 else (val >> shift) | lower64((val << (64 - shift)))

def rotateByAtleast1(val, shift):
    """
    Bitwise right rotate, same as rotate, but requires shift to be non-zero.
    """
    return (val >> shift) | lower64((val << (64 - shift)))

def shiftMix(val):
    """
    """
    return lower64(val ^ (val >> 47))

def hashLen16(u, v):
    """
    """
    uv = (v << 64) | u
    return hash128to64(uv)

def hashLen0to16(candidate):
    length = len(candidate)
    if length > 8:
        a = bytes(candidate[0:8])
        b = bytes(candidate[-8:-1] + candidate[-1])
        return hashLen16(a, rotateByAtleast1(b + length, length)) ^ b
    if length >= 4:
        a = bytes(candidate[0:4])
        return hashLen16(length + (a << 3), bytes(candidate[-4:-1] + candidate[-1]))
    if length > 0:
        a = bytes(candidate[0])
        b = bytes(candidate[length >> 1])
        c = bytes(candidate[length - 1])
        y = lower32(a + (b << 8))
        z = length + c * 4
        return lower64(shiftMix(lower64(y * K2 ^ z * K3)) * K2)
    return K2 # Changed before was k2

def hashLen17To32(candidate):
    a = lower64(bytes(candidate[0:8]) * K1)
    b = bytes(candidate[8:16])
    c = lower64(bytes(candidate[-8:-1] + candidate[-1]) * K2)
    d = lower64(bytes(candidate[-16:-8]) * K0)
    return hashLen16(lower64(rotate(lower64(a - b), 43) + rotate(c, 30) + d),
                     lower64(a + rotate(b ^ K3, 30) - c) + len(candidate))

def _weakHashLen32WithSeeds(w, x, y, z, a, b):
    a += w
    b = rotate(lower64(b + a + z), 21)
    c = a
    a += x
    a = lower64(a + y)
    b += rotate(a, 44)
    return lower64(a + z) << 64 | lower64(b + c)

def weakHashLen32WithSeeds(candidate, a, b):
    return _weakHashLen32WithSeeds(bytes(candidate[0:8]),
                                   bytes(candidate[8:16]),
                                   bytes(candidate[16:24]),
                                   bytes(candidate[24:32]),
                                   a,
                                   b)

def hashLen33To64(candidate):
    """
    """
    length = len(candidate)
    z = bytes(candidate[24:32])
    a = bytes(candidate[0:8]) + (length + bytes(candidate[-16:-8])) * K0
    a = lower64(a)
    b = rotate(lower64(a + z), 52)
    c = rotate(a, 37)
    a = lower64(a + bytes(candidate[8:16]))
    c = lower64(c + rotate(a, 7))
    a = lower64(a + bytes(candidate[16:24]))
    vf = lower64(a + z)
    vs = lower64(b + rotate(a, 31) + c)
    a = bytes(candidate[16: 24]) + bytes(candidate[-32:-24])
    z = bytes(candidate[-8:-1] + candidate[-1])
    b = rotate(lower64(a + z), 52)
    c = rotate(a, 37)
    a = lower64(a + bytes(candidate[-24:-16]))
    c = lower64(c + rotate(a, 7))
    a = lower64(a + bytes(candidate[-16:-8]))
    wf = lower64(a + z)
    ws = lower64(b + rotate(a, 31) + c)
    r = shiftMix(lower64((vf + ws) * K2 + (wf + vs) + K0))
    return lower64(shiftMix(lower64(r * K0 + vs)) * K2)

def hashLenAbove64(candidate):
    """
    """
    length = len(candidate)
    x = bytes(candidate[0:8])
    y = bytes(candidate[-16:-8]) ^ K1
    z = bytes(candidate[-56:-48]) ^ K0
    v = weakHashLen32WithSeeds(candidate[-64:-1] + candidate[-1], length, y)
    w = weakHashLen32WithSeeds(candidate[-32:-1] + candidate[-1], lower64(length * K1), K0)

    z = lower64(z + shiftMix(lower64(v)) * K1)
    x = lower64(rotate(lower64(z + x), 39) * K1)
    y = lower64(rotate(y, 33) * K1)

    length = (length - 1) & ~63
    while length is not 0:
        xrv = lower64(x + y + higher64(v) + bytes(candidate[16:24]))
        yrv = lower64(y + lower64(v) + bytes(candidate[48:56]))
        x = lower64(rotate(xrv, 37) * K1)
        y = lower64(rotate(yrv, 42) * K1)
        x = lower64(rotate(xrv, 37) * K1)
        x ^= lower64(w)
        y ^= lower64(v)
        z = rotate(z ^ higher64(w), 33)
        v = weakHashLen32WithSeeds(candidate,
                                   lower64(lower64(v) * K1),
                                   lower64(x + higher64(w)))
        w = weakHashLen32WithSeeds(candidate[32:-1] + candidate[-1],
                                   lower64(z + lower64(w)),
                                   y)
        z, x = x, z
        candidate = candidate[64:-1] + candidate[-1]
        length -= 64

    return hashLen16(lower64(hashLen16(higher64(v), higher64(w)) +
                             shiftMix(y) *
                             K1 +
                             z),
                     lower64(hashLen16(lower64(v), lower64(w)) +
                             x))

def cityMurmur(candidate, seed):
    """
    """
    length = len(candidate)
    a = lower64(seed)
    b = higher64(seed)
    c, d = 0, 0
    l = length - 16
    if l <= 0:
        a = lower64(shiftMix(lower64(a * K1)) * K1)
        c = lower64(b * K1 + hashLen0to16(candidate))
        d = shiftMix(lower64(a + (bytes(candidate[0:8]) if length >= 8 else c)))
    else:
        c = hashLen16(lower64(bytes(candidate[-8:-1] + candidate[-1]) + K1), a)
        d = hashLen16(lower64(b + length), lower64(c + bytes(candidate[-16:-8])))
        a = lower64(a + d)
        while l > 0:
            a ^= lower64(shiftMix(lower64(bytes(candidate[0:8]) * K1)) * K1)
            a = lower64(a * K1)
            b ^= a
            c ^= lower64(shiftMix(lower64(bytes(candidate[8:16]) * K1)) * K1)
            c = lower64(c * K1)
            d ^= c
            candidate = candidate[16:-1] + candidate[-1]
            l -= 16
    a = hashLen16(a, c)
    b = hashLen16(d, b)

    return ((a ^ b) << 64) | hashLen16(b, a)

def hash128WithSeed(candidate, seed):
    originalCandidate = candidate
    length = len(candidate)
    if length < 128:
        return cityMurmur(candidate, seed)
    else:
        x = lower64(seed)
        y = higher64(seed)
        z = lower64(length * K1)

        vf = lower64(lower64(rotate(y ^ K1, 49) * K1) + bytes(candidate[0:8]))
        vs = lower64(lower64(rotate(vf, 42) * K1) + bytes(candidate[8:16]))
        wf = lower64(lower64(rotate(lower64(y + z), 35) * K1) + x)
        ws = lower64(rotate(lower64(x + bytes(candidate[88:96])), 53) * K1)
        v = (vf << 64) | vs
        w = (wf << 64) | ws

        while length >= 128:
            x = lower64(rotate(lower64(x + y + vf + bytes(candidate[16:24])), 37) * K1)
            y = lower64(rotate(lower64(y + vs + bytes(candidate[48:55])), 42) * K1)
            x ^= ws
            y ^= vf
            z = rotate(z ^ wf, 33)
            v = weakHashLen32WithSeeds(candidate,
                                       lower64(z + ws),
                                       lower64(x + wf))
            w = weakHashLen32WithSeeds(candidate[32:-1] + candidate[-1],
                                       lower64(z + ws),
                                       y)
            vf, vs = higher64(v), lower64(v)
            wf, ws = higher64(w), lower64(w)
            z, x = x, z
            candidate = candidate[64:-1] + candidate[-1]

            x = lower64(rotate(lower64(x + y + vf + bytes(candidate[16:24])), 37) * K1)
            y = lower64(rotate(lower64(y + vs + bytes(candidate[48:56])), 42) * K1)
            z = rotate(z ^ wf, 33)

            v = weakHashLen32WithSeeds(candidate,
                                       lower64(vs * K1),
                                       lower64(x + wf))
            w = weakHashLen32WithSeeds(candidate[32:-1] + candidate[-1],
                                       lower64(z + ws),
                                       y)
            vf, vs = higher64(v), lower64(v)
            wf, ws = higher64(w), lower64(w)
            z, x = x, z
            candidate = candidate[64:-1] + candidate[-1]
            length -= 128

        y = lower64(y + rotate(wf, 37) * K0 + z)
        x = lower64(x + rotate(lower64(vf + z), 49) * K0)

        tail_done = 0

        while tail_done < length:
            tail_done += 32
            y = lower64(rotate(lower64(y - x), 42) * K0 + vs)
            wf = lower64(wf + bytes(originalCandidate[16 - tail_done:24 - tail_done]))
            x = lower64(rotate(x, 49) * K0 + wf)
            wf = lower64(wf + vf)
            v = weakHashLen32WithSeeds(originalCandidate[-tail_done:-1] + originalCandidate[-1],
                                       vf,
                                       vs)
            vf, vs = higher64(v), lower64(v)

        x = hashLen16(x, vf)
        y = hashLen16(y, wf)
        hf = lower64(hashLen16(lower64(x + vs), ws) + y)
        hs = lower64(hashLen16(lower64(x + ws), lower64(y + vs)))

        return (hf << 64) | hs

def hash64(candidate):
    length = len(candidate)

    if length <= 16:
        return hashLen0to16(candidate)
    elif length <= 32:
        return hashLen17To32(candidate)
    elif length <= 64:
        return hashLen33To64(candidate)
    else:
        return hashLenAbove64(candidate)

def hash64WithSeeds(candidate, seed0, seed1):
    return hashLen16(lower64(hash64(candidate) - seed0), seed1)

def hash64WithSeed(candidate, seed):
    return hash64WithSeeds(candidate, K2, seed)

def hash128(candidate):
    length = len(candidate)

    if length >= 16:
        seed = ((bytes(candidate[8:16]) << 64) | (bytes(candidate[0:8]) ^ K3))
        return hash128WithSeed(candidate[16:-1] + candidate[-1], seed)
    else:
        return hash128WithSeed(candidate, (K1 << 64) | K0)