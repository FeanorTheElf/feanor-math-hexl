use cpp::cpp;
use feanor_math::algorithms::fft::cooley_tuckey::bitreverse;
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
        // the standard convention for fourier transform is to perform the forward transform with `𝝵^-1`,
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

///
/// Wrapper around a `intel::hexl::NTT` object that can compute the negacyclic NTT of
/// fixed (power-of-two) length over a fixed ring.
/// 
pub struct HEXLNegacyclicNTT {
    data: *mut c_void,
    ring: Zn,
    log2_n: usize,
    root_of_unity: El<Zn>
}

impl HEXLNegacyclicNTT {

    ///
    /// Creates a new HEXL NTT object.
    /// 
    /// Note that this follows the usual convention and performs the forward transform with `𝝵^-1` where
    /// `𝝵` is the given root of unity. In other words, the forward transform computes
    /// ```text
    ///  (a_i) -> ( sum a_i 𝝵^(-ij) )_j
    /// ```
    /// where `j` is a unit in `Z/2nZ`. This is different from the convention in hexl, where the forward transform
    /// uses `𝝵` and not `𝝵^-1`.
    /// 
    pub fn new(ring: Zn, log2_n: usize, root_of_unity: El<Zn>) -> Self {
        assert!(is_prim_root_of_unity_pow2(ring, &root_of_unity, log2_n + 1));
        let q: u64 = *ring.modulus() as u64;
        let n: u64 = (1 << log2_n) as u64;
        let inv_root_of_unity_scalar = ring.smallest_positive_lift(ring.invert(&root_of_unity).unwrap()) as u64;
        let pointer = unsafe { cpp!( [n as "uint64_t", q as "uint64_t", inv_root_of_unity_scalar as "uint64_t"] -> *mut c_void as "void*" { return create_ntt(n, q, inv_root_of_unity_scalar); } ) };
        HEXLNegacyclicNTT { data: pointer, ring: ring, log2_n: log2_n, root_of_unity: root_of_unity }
    }

    pub fn for_zn(ring: Zn, log2_n: usize) -> Option<Self> {
        let as_field = (&ring).as_field().ok().unwrap();
        let root_of_unity = as_field.get_ring().unwrap_element(get_prim_root_of_unity_pow2(as_field, log2_n + 1)?);
        return Some(Self::new(ring, log2_n, root_of_unity));
    }

    ///
    /// On input `i`, returns `j` such that `unordered_negacyclic_fft_base(values)[i]` contains the
    /// evaluation at `𝝵^(-j)`, where `𝝵` is the root of unity returned by [`HEXLNegacyclicNTT::root_of_unity()`].
    /// 
    pub fn unordered_negacyclic_fft_permutation(&self, i: usize) -> usize {
        assert!(i < (1 << self.log2_n));
        bitreverse(i, self.log2_n + 1) + 1
    }

    ///
    /// Inverse of [`HEXLNegacyclicNTT::unordered_negacyclic_fft_permutation()`].
    /// 
    pub fn unordered_negacyclic_fft_inv_permutation(&self, j: usize) -> usize {
        assert!(j % 2 == 1);
        bitreverse(j - 1, self.log2_n + 1)
    }

    ///
    /// The elements of input won't change their value, but might be reduced in-place.
    /// 
    pub fn unordered_negacyclic_fft_base<const INV: bool>(&self, input: &mut [ZnEl], output: &mut [ZnEl]) {
        assert_eq!(1 << self.log2_n, input.len());
        assert_eq!(1 << self.log2_n, output.len());
        assert_eq!(std::mem::size_of::<ZnEl>(), std::mem::size_of::<u64>());
        assert_eq!(std::mem::align_of::<ZnEl>(), std::mem::align_of::<u64>());
        let len = input.len();
        let input_znel_ptr = input.as_mut_ptr();
        let input_u64_ptr = input_znel_ptr as *mut u64;
        let output_znel_ptr = output.as_mut_ptr();
        let output_u64_ptr = output_znel_ptr as *mut u64;
        for i in 0..len {
            // Safety: both `input_znel_ptr` and `input_u64_ptr` are valid pointers to the same array;
            // since they are raw pointers, it is ok that they alias
            unsafe {
                let x = std::ptr::read(input_znel_ptr.offset(i as isize));
                let x_as_u64 = self.ring.smallest_positive_lift(x) as u64;
                std::ptr::write(input_u64_ptr.offset(i as isize), x_as_u64);
            }
        }
        {
            let ptr = self.data;
            if INV {
                unsafe { cpp!( [ptr as "void*", input_u64_ptr as "const uint64_t*", output_u64_ptr as "uint64_t*"] { ntt_backward(static_cast<intel::hexl::NTT*>(ptr), input_u64_ptr, output_u64_ptr); } ) }
            } else {
                unsafe { cpp!( [ptr as "void*", input_u64_ptr as "const uint64_t*", output_u64_ptr as "uint64_t*"] { ntt_forward(static_cast<intel::hexl::NTT*>(ptr), input_u64_ptr, output_u64_ptr); } ) }
            }
        }
        for i in 0..len {
            // Safety: both `output_znel_ptr` and `output_u64_ptr` are valid pointers to the same array;
            // since they are raw pointers, it is ok that they alias
            unsafe {
                let x = std::ptr::read(output_u64_ptr.offset(i as isize));
                let x_as_znel = self.ring.get_ring().from_int_promise_reduced(x as i64);
                std::ptr::write(output_znel_ptr.offset(i as isize), x_as_znel);
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

///
/// The github page says that the "Intel HE Acceleration Library is single-threaded and thread-safe."
/// 
unsafe impl Send for HEXLNegacyclicNTT {}

///
/// The github page says that the "Intel HE Acceleration Library is single-threaded and thread-safe."
/// 
unsafe impl Sync for HEXLNegacyclicNTT {}

impl PartialEq for HEXLNegacyclicNTT {

    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring() && self.ring.eq_el(&self.root_of_unity, &other.root_of_unity)
    }
}

impl Drop for HEXLNegacyclicNTT {

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
fn test_ntt_length_4() {
    let ring = Zn::new(17);
    let z = ring.int_hom().map(9);
    let inv_z = ring.invert(&z).unwrap();
    let fft = HEXLNegacyclicNTT::new(ring, 2, z);
    assert_el_eq!(&ring, z, fft.root_of_unity());
    let one = fft.ring().one();
    let zero = fft.ring().zero();

    {
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
}

#[test]
fn test_ntt_length_8() {
    let ring = Zn::new(17);
    let z = ring.int_hom().map(3);
    let fft = HEXLNegacyclicNTT::new(ring, 3, z);
    let one = fft.ring().one();
    let zero = fft.ring().zero();

    let mut values = [zero, one, one, zero, zero, zero, zero, zero];
    let mut original = values;

    // bitreversed order
    let mut expected = [zero; 8];

    for i in 0..8 {
        let power_of_z = ring.pow(ring.invert(fft.root_of_unity()).unwrap(), fft.unordered_negacyclic_fft_permutation(i));
        expected[i] = ring.add(power_of_z, ring.pow(power_of_z, 2));
    }

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

#[cfg(test)]
fn run_fft_bench_round(fft: &HEXLNegacyclicNTT, data: &mut [ZnEl], ntt: &mut [ZnEl], inv_ntt: &mut [ZnEl]) {
    fft.unordered_negacyclic_fft_base::<false>(data, ntt);
    fft.unordered_negacyclic_fft_base::<true>(ntt, inv_ntt);
    assert_el_eq!(fft.ring(), data[0], inv_ntt[0]);
}

#[cfg(test)]
const BENCH_SIZE_LOG2: usize = 16;

#[bench]
fn bench_fft_zn_64(bencher: &mut Bencher) {
    let ring = Zn::new(1073872897);
    let fft = HEXLNegacyclicNTT::for_zn(ring.clone(), BENCH_SIZE_LOG2).unwrap();
    let mut data = (0..(1 << BENCH_SIZE_LOG2)).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
    let mut ntt = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    ntt.resize_with(1 << BENCH_SIZE_LOG2, || fft.ring().zero());
    let mut inv_ntt = Vec::with_capacity(1 << BENCH_SIZE_LOG2);
    inv_ntt.resize_with(1 << BENCH_SIZE_LOG2, || fft.ring().zero());
    bencher.iter(|| {
        run_fft_bench_round(&fft, &mut data, &mut ntt, &mut inv_ntt);
    });
}