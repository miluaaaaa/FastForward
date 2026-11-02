/* RGB8, non-interlaced PNG row reconstruction. Public PNG filter arithmetic;
 * private implementation, no changes to libpng/libspng/shared environments.
 * Buffers must have 16 extra accessible bytes; sizes are checked by Python. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>

static inline int paeth(int a,int b,int c) {
    int p=a+b-c,pa=p-a,pb=p-b,pc=p-c;
    if(pa<0)pa=-pa;if(pb<0)pb=-pb;if(pc<0)pc=-pc;
    return pa<=pb && pa<=pc ? a : pb<=pc ? b : c;
}

int reconstruct_rgb(const uint8_t *input,size_t width,size_t height,uint8_t *out,int simd) {
    const size_t n=width*3;
    if(!width || !height || width>32768 || height>32768)return -1;
    __m128i zero=_mm_setzero_si128();
    __m128i pattern=_mm_setr_epi8(13,14,15,13,14,15,13,14,15,13,14,15,13,14,15,13);
    for(size_t y=0;y<height;y++) {
        const uint8_t *raw=input+y*(n+1)+1;
        uint8_t *row=out+y*n;
        const uint8_t *above=y ? row-n : NULL;
        int filter=raw[-1];size_t i=0;
        if(filter==0) {memcpy(row,raw,n);continue;}
        if(filter<1 || filter>4)return -2;
        if(simd && filter==1) {
            __m128i carry=zero;
            for(;i+16<=n;i+=16) {
                __m128i x=_mm_loadu_si128((const __m128i*)(raw+i));
                x=_mm_add_epi8(x,_mm_slli_si128(x,3));
                x=_mm_add_epi8(x,_mm_slli_si128(x,6));
                x=_mm_add_epi8(x,_mm_slli_si128(x,12));
                x=_mm_add_epi8(x,carry);
                _mm_storeu_si128((__m128i*)(row+i),x);
                carry=_mm_shuffle_epi8(x,pattern);
            }
        } else if(simd && filter==2) {
            if(!above){memcpy(row,raw,n);continue;}
            for(;i+16<=n;i+=16) {
                __m128i x=_mm_loadu_si128((const __m128i*)(raw+i));
                __m128i b=_mm_loadu_si128((const __m128i*)(above+i));
                _mm_storeu_si128((__m128i*)(row+i),_mm_add_epi8(x,b));
            }
        } else if(simd && (filter==3 || filter==4)) {
            __m128i left=zero,upperleft=zero;
            for(;i<n;) {
                /* If left equals upper-left, a zero residual reconstructs
                 * the exact pixel above. That invariant propagates through
                 * an arbitrary zero run, even when the row above varies. */
                if(simd>1 && filter==4 && above && i+48<=n &&
                   ((_mm_cvtsi128_si32(left)^_mm_cvtsi128_si32(upperleft))&0xffffff)==0) {
                    __m128i r0=_mm_loadu_si128((const __m128i*)(raw+i));
                    __m128i r1=_mm_loadu_si128((const __m128i*)(raw+i+16));
                    __m128i r2=_mm_loadu_si128((const __m128i*)(raw+i+32));
                    __m128i bits=_mm_or_si128(_mm_or_si128(r0,r1),r2);
                    if(_mm_testz_si128(bits,bits)) {
                        _mm_storeu_si128((__m128i*)(row+i),_mm_loadu_si128((const __m128i*)(above+i)));
                        _mm_storeu_si128((__m128i*)(row+i+16),_mm_loadu_si128((const __m128i*)(above+i+16)));
                        _mm_storeu_si128((__m128i*)(row+i+32),_mm_loadu_si128((const __m128i*)(above+i+32)));
                        uint32_t last;memcpy(&last,above+i+45,4);
                        left=upperleft=_mm_cvtsi32_si128((int)last);i+=48;continue;
                    }
                }
                uint32_t r,b=0;memcpy(&r,raw+i,4);if(above)memcpy(&b,above+i,4);
                __m128i source=_mm_cvtsi32_si128((int)r),up=_mm_cvtsi32_si128((int)b);
                __m128i pred;
                if(filter==3) {
                    __m128i odd=_mm_and_si128(_mm_xor_si128(left,up),_mm_set1_epi8(1));
                    pred=_mm_sub_epi8(_mm_avg_epu8(left,up),odd);
                } else {
                    __m128i a=_mm_unpacklo_epi8(left,zero),bb=_mm_unpacklo_epi8(up,zero);
                    __m128i c=_mm_unpacklo_epi8(upperleft,zero);
                    __m128i pa=_mm_abs_epi16(_mm_sub_epi16(bb,c));
                    __m128i pb=_mm_abs_epi16(_mm_sub_epi16(a,c));
                    __m128i pc=_mm_abs_epi16(_mm_sub_epi16(_mm_add_epi16(a,bb),_mm_slli_epi16(c,1)));
                    __m128i choosea=_mm_and_si128(_mm_cmpeq_epi16(_mm_min_epi16(pa,pb),pa),
                                                _mm_cmpeq_epi16(_mm_min_epi16(pa,pc),pa));
                    __m128i chooseb=_mm_cmpeq_epi16(_mm_min_epi16(pb,pc),pb);
                    pred=_mm_blendv_epi8(c,bb,chooseb);pred=_mm_blendv_epi8(pred,a,choosea);
                    pred=_mm_packus_epi16(pred,zero);
                }
                left=_mm_add_epi8(source,pred);upperleft=up;
                r=(uint32_t)_mm_cvtsi128_si32(left);memcpy(row+i,&r,4);
                i+=3;
            }
        }
        for(;i<n;i++) {
            int a=i>=3?row[i-3]:0,b=above?above[i]:0,c=above && i>=3?above[i-3]:0;
            int predict=filter==1?a:filter==2?b:filter==3?(a+b)/2:paeth(a,b,c);
            row[i]=(uint8_t)(raw[i]+predict);
        }
    }
    return 0;
}
