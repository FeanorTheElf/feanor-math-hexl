use cpp::cpp;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::ring::{El, RingStore};
use feanor_math::rings::zn::zn_64::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::algorithms::unity_root::{get_prim_root_of_unity_pow2, is_prim_root_of_unity_pow2};

use std::ffi::c_void;

cpp!{{
    #include "hexl/ntt/ntt.hpp"

    intel::hexl::NTT* create_ntt(uint64_t N, uint64_t modulus, uint64_t inv_root_of_unity) {
        // the standard convention for fourier transform is to perform the forward transform with z^-1,
        // but hexl does the forward transform with the passed root of unity
        return new intel::hexl::NTT(N, modulus, inv_root_of_unity);
    }

    void destroy_ntt(intel::hexl::NTT* ntt) {
        delete ntt;
    }

    void ntt_forward(intel::hexl::NTT* ntt, const uint64_t* in, uint64_t* out) {
        ntt->ComputeForward(out, in, 1, 1);
    }

    uint64_t get_root_of_unity(intel::hexl::NTT* ntt) {
        return ntt->GetMinimalRootOfUnity();
    }

    void ntt_backward(intel::hexl::NTT* ntt, const uint64_t* in, uint64_t* out) {
        ntt->ComputeInverse(out, in, 1, 1);
    }
}}

pub struct HEXLNTT {
    data: *mut c_void,
    ring: Zn,
    log2_n: usize,
    root_of_unity: El<Zn>
}

impl HEXLNTT {

    ///
    /// Creates a new HEXL NTT object.
    /// 
    /// Note that this follows the usual convention and performs the forward transform with `z^-1` where
    /// `z` is the given root of unity. In other words, the forward transform computes
    /// ```text
    ///  (a_i) -> ( sum a_i z^(-i * j) )_j
    /// ```
    /// where `j` is a unit in `Z/2NZ`. This is different from the convention in hexl, where the forward transform
    /// uses `z` and not `z^-1`.
    /// 
    pub fn new(ring: Zn, log2_n: usize, root_of_unity: El<Zn>) -> Self {
        assert!(is_prim_root_of_unity_pow2(ring, &root_of_unity, log2_n + 1));
        let q: u64 = *ring.modulus() as u64;
        let n: u64 = (1 << log2_n) as u64;
        let inv_root_of_unity_scalar = ring.smallest_positive_lift(ring.invert(&root_of_unity).unwrap()) as u64;
        let pointer = unsafe { cpp!( [n as "uint64_t", q as "uint64_t", inv_root_of_unity_scalar as "uint64_t"] -> *mut c_void as "void*" { return create_ntt(n, q, inv_root_of_unity_scalar); } ) };
        HEXLNTT { data: pointer, ring: ring, log2_n: log2_n, root_of_unity: root_of_unity }
    }

    pub fn for_zn(ring: Zn, log2_n: usize) -> Option<Self> {
        let as_field = (&ring).as_field().ok().unwrap();
        let root_of_unity = as_field.get_ring().unwrap_element(get_prim_root_of_unity_pow2(as_field, log2_n + 1)?);
        return Some(Self::new(ring, log2_n, root_of_unity));
    }

    ///
    /// The elements of input won't change their value, but might be reduced in-place.
    /// 
    pub fn unordered_negacyclic_fft_base<const INV: bool>(&self, input: &mut [ZnEl], output: &mut [ZnEl]) {
        assert_eq!(1 << self.log2_n, input.len());
        assert_eq!(1 << self.log2_n, output.len());
        assert_eq!(std::mem::size_of::<ZnEl>(), std::mem::size_of::<u64>());
        assert_eq!(std::mem::align_of::<ZnEl>(), std::mem::align_of::<u64>());
        for x in &mut*input {
            *x = self.ring.get_ring().from_int_promise_reduced(self.ring.smallest_positive_lift(*x));
        }
        {
            let ptr = self.data;
            let input_ptr = unsafe { std::mem::transmute::<*const ZnEl, *const u64>(input.as_ptr()) };
            let output_ptr = unsafe { std::mem::transmute::<*mut ZnEl, *mut u64>(output.as_mut_ptr()) };
            if INV {
                unsafe { cpp!( [ptr as "void*", input_ptr as "const uint64_t*", output_ptr as "uint64_t*"] { ntt_backward(static_cast<intel::hexl::NTT*>(ptr), input_ptr, output_ptr); } ) }
            } else {
                unsafe { cpp!( [ptr as "void*", input_ptr as "const uint64_t*", output_ptr as "uint64_t*"] { ntt_forward(static_cast<intel::hexl::NTT*>(ptr), input_ptr, output_ptr); } ) }
            }
        }
    }

    pub fn ring(&self) -> &Zn {
        &self.ring
    }

    pub fn root_of_unity(&self) -> &El<Zn> {
        &self.root_of_unity
    }

    pub fn n(&self) -> usize {
        1 << self.log2_n
    }
}

impl PartialEq for HEXLNTT {
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring() && self.ring.eq_el(&self.root_of_unity, &other.root_of_unity)
    }
}

impl Drop for HEXLNTT {

    fn drop(&mut self) {
        let ptr = self.data;
        unsafe { cpp!( [ptr as "void*"] { destroy_ntt(static_cast<intel::hexl::NTT*>(ptr)); } ) }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use test::Bencher;
#[cfg(test)]
use feanor_math::homomorphism::Homomorphism;

#[test]
fn test_ntt() {
    let ring = Zn::new(17);

    // length 4
    {
        let z = ring.int_hom().map(9);
        let fft = HEXLNTT::new(ring, 2, z);
        let one = fft.ring().one();
        let zero = fft.ring().zero();

        let mut values = [one, zero, zero, zero];
        let mut original = values;

        // bitreversed order
        let expected = [one, one, one, one];

        fft.unordered_negacyclic_fft_base::<false>(&mut original, &mut values);
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &expected[i], &values[i]);
        }

        let mut result = [zero; 4];
        fft.unordered_negacyclic_fft_base::<true>(&mut values, &mut result);
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &original[i], &result[i]);
        }
    }
    {
        let z = ring.int_hom().map(9);
        let inv_z = ring.invert(&z).unwrap();
        let fft = HEXLNTT::new(ring, 2, z);
        let one = fft.ring().one();
        let zero = fft.ring().zero();

        let mut values = [zero, one, zero, zero];
        let mut original = values;

        // bitreversed order
        let expected = [inv_z, ring.pow(inv_z, 5), ring.pow(inv_z, 3), ring.pow(inv_z, 7)];

        fft.unordered_negacyclic_fft_base::<false>(&mut original, &mut values);
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &expected[i], &values[i]);
        }

        let mut result = [zero; 4];
        fft.unordered_negacyclic_fft_base::<true>(&mut values, &mut result);
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &original[i], &result[i]);
        }
    }
    // length 8
    {
        let z = ring.int_hom().map(3);
        let inv_z = ring.invert(&z).unwrap();
        let fft = HEXLNTT::new(ring, 3, z);
        let one = fft.ring().one();
        let zero = fft.ring().zero();

        let mut values = [zero, one, one, zero, zero, zero, zero, zero];
        let mut original = values;

        // bitreversed order
        let expected = [
            ring.add(inv_z, ring.pow(inv_z, 2)),
            ring.add(ring.pow(inv_z, 9), ring.pow(inv_z, 18)),
            ring.add(ring.pow(inv_z, 5), ring.pow(inv_z, 10)),
            ring.add(ring.pow(inv_z, 13), ring.pow(inv_z, 26)),
            ring.add(ring.pow(inv_z, 3), ring.pow(inv_z, 6)),
            ring.add(ring.pow(inv_z, 11), ring.pow(inv_z, 22)),
            ring.add(ring.pow(inv_z, 7), ring.pow(inv_z, 14)),
            ring.add(ring.pow(inv_z, 15),ring.pow(inv_z, 30))
        ];

        fft.unordered_negacyclic_fft_base::<false>(&mut original, &mut values);
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &expected[i], &values[i]);
        }

        let mut result = [zero; 8];
        fft.unordered_negacyclic_fft_base::<true>(&mut values, &mut result);
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &original[i], &result[i]);
        }
    }
}

#[cfg(test)]
fn run_fft_bench_round(fft: &HEXLNTT, data: &mut [ZnEl], ntt: &mut [ZnEl], inv_ntt: &mut [ZnEl]) {
    fft.unordered_negacyclic_fft_base::<false>(data, ntt);
    fft.unordered_negacyclic_fft_base::<true>(ntt, inv_ntt);
    assert_el_eq!(fft.ring(), data[0], inv_ntt[0]);
}

#[cfg(test)]
const BENCH_SIZE_LOG2: usize = 16;

#[bench]
fn bench_fft_zn_64(bencher: &mut Bencher) {
    let ring = Zn::new(1073872897);
    let fft = HEXLNTT::for_zn(ring.clone(), BENCH_SIZE_LOG2).unwrap();
    let mut data = (0..(1 << BENCH_SIZE_LOG2)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    let mut ntt = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    ntt.resize_with(1 << BENCH_SIZE_LOG2, || fft.ring().zero());
    let mut inv_ntt = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    inv_ntt.resize_with(1 << BENCH_SIZE_LOG2, || fft.ring().zero());
    bencher.iter(|| {
        run_fft_bench_round(&fft, &mut data, &mut ntt, &mut inv_ntt);
    });
}