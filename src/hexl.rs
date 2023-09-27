use cpp::cpp;

use feanor_math::algorithms::fft::cooley_tuckey::bitreverse;
use feanor_math::algorithms::unity_root;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::rings::zn::zn_42::Zn;
use feanor_math::rings::zn::*;
use feanor_math::integer::IntegerRingStore;
use feanor_math::mempool::{MemoryProvider, AllocatingMemoryProvider};
use feanor_math::ring::*;
use feanor_math::primitive_int::*;
use feanor_math::vector::*;

use std::ffi::c_void;
use std::ops::DerefMut;

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

struct HEXLNTT<M: MemoryProvider<u64> = AllocatingMemoryProvider> {
    data: *mut c_void,
    ring: Zn,
    len: u64,
    memory_provider: M,
    root_of_unity: El<Zn>
}

impl<M: MemoryProvider<u64>> HEXLNTT<M> {

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
    pub fn new(n: u64, ring: Zn, root_of_unity: El<Zn>, memory_provider: M) -> Self {
        // N must be power of two
        let log2_n = StaticRing::<i64>::RING.abs_log2_ceil(&(n as i64)).unwrap();
        assert!((1 << log2_n) == n);
        assert!(unity_root::is_prim_root_of_unity_pow2(ring, &root_of_unity, log2_n + 1));
        let q = *ring.modulus() as u64;
        let inv_root_of_unity_scalar = ring.smallest_positive_lift(ring.invert(&root_of_unity).unwrap()) as u64;
        let pointer = unsafe { cpp!( [n as "uint64_t", q as "uint64_t", inv_root_of_unity_scalar as "uint64_t"] -> *mut c_void as "void*" { return create_ntt(n, q, inv_root_of_unity_scalar); } ) };
        HEXLNTT { data: pointer, ring: ring, len: n, memory_provider: memory_provider, root_of_unity: root_of_unity }
    }

    fn log2_n(&self) -> usize {
        StaticRing::<i64>::RING.abs_log2_ceil(&(self.len() as i64)).unwrap()
    }

    fn out_of_place_negacyclic_fft<S, const INV: bool>(&self, mut input: M::Object, ring: S) -> M::Object
        where S: RingStore, 
            S::Type: CanonicalIso<<Zn as RingStore>::Type>
    {
        let mut output = self.memory_provider.get_new_init(self.len as usize, |_| 0);
        {
            let ptr = self.data;
            let input_ptr = input.deref_mut().as_ptr();
            let output_ptr = output.deref_mut().as_ptr();
            if INV {
                unsafe { cpp!( [ptr as "void*", input_ptr as "const uint64_t*", output_ptr as "uint64_t*"] { ntt_backward(static_cast<intel::hexl::NTT*>(ptr), input_ptr, output_ptr); } ) }
            } else {
                unsafe { cpp!( [ptr as "void*", input_ptr as "const uint64_t*", output_ptr as "uint64_t*"] { ntt_forward(static_cast<intel::hexl::NTT*>(ptr), input_ptr, output_ptr); } ) }
            }
        }
        return output;
    }
}

impl<M: MemoryProvider<u64>> Drop for HEXLNTT<M> {

    fn drop(&mut self) {
        let ptr = self.data;
        unsafe { cpp!( [ptr as "void*"] { destroy_ntt(static_cast<intel::hexl::NTT*>(ptr)); } ) }
    }
}

impl<M: MemoryProvider<u64>> HEXLNTT<M> {

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn ring(&self) -> &Zn {
        &self.ring
    }

    pub fn unordered_negacyclic_ntt<V, S, M2>(&self, mut values: V, ring: S, _memory_provider: &M2)
        where S: RingStore, 
            S::Type: CanonicalIso<<Zn as RingStore>::Type>, 
            V: VectorViewMut<feanor_math::ring::El<S>>,
            M2: MemoryProvider<feanor_math::ring::El<S>>
    {
        let iso = ring.can_iso(self.ring()).unwrap();
        let input = self.memory_provider.get_new_init(self.len as usize, |i| self.ring().smallest_positive_lift(iso.map_back(ring.clone_el(values.at(i)))) as u64);
        let output = self.out_of_place_negacyclic_fft::<_, false>(input, &ring);
        for i in 0..(self.len as usize) {
            *values.at_mut(i) = iso.map(self.ring().coerce(&StaticRing::<i64>::RING, output[i] as i64));
        }
    }

    pub fn negacyclic_ntt<V, S, M2>(&self, mut values: V, ring: S, _memory_provider: &M2) 
        where S: RingStore, 
            S::Type: CanonicalIso<<Zn as RingStore>::Type>, 
            V: VectorViewMut<feanor_math::ring::El<S>>,
            M2: MemoryProvider<feanor_math::ring::El<S>>
    {
        let iso = ring.can_iso(self.ring()).unwrap();
        let input = self.memory_provider.get_new_init(self.len as usize, |i| self.ring().smallest_positive_lift(iso.map_back(ring.clone_el(values.at(i)))) as u64);
        let output = self.out_of_place_negacyclic_fft::<_, false>(input, &ring);
        for i in 0..(self.len as usize) {
            *values.at_mut(bitreverse(i, self.log2_n())) = iso.map(self.ring().coerce(&StaticRing::<i64>::RING, output[i] as i64));
        }
    }

    pub fn unordered_negacyclic_inv_ntt<V, S, M2>(&self, mut values: V, ring: S, _memory_provider: &M2)
        where S: RingStore, 
            S::Type: CanonicalIso<<Zn as RingStore>::Type>, 
            V: VectorViewMut<feanor_math::ring::El<S>>,
            M2: MemoryProvider<feanor_math::ring::El<S>>
    {
        let iso = ring.can_iso(self.ring()).unwrap();
        let input = self.memory_provider.get_new_init(self.len as usize, |i| self.ring().smallest_positive_lift(iso.map_back(ring.clone_el(values.at(i)))) as u64);
        let output = self.out_of_place_negacyclic_fft::<_, true>(input, &ring);
        for i in 0..(self.len as usize) {
            *values.at_mut(i) = iso.map(self.ring().coerce(&StaticRing::<i64>::RING, output[i] as i64));
        }
    }

    pub fn negacyclic_inv_ntt<V, S, M2>(&self, mut values: V, ring: S, _memory_provider: &M2) 
        where S: RingStore, 
            S::Type: CanonicalIso<<Zn as RingStore>::Type>, 
            V: VectorViewMut<feanor_math::ring::El<S>>,
            M2: MemoryProvider<feanor_math::ring::El<S>>
    {
        let iso = ring.can_iso(self.ring()).unwrap();
        let input = self.memory_provider.get_new_init(self.len as usize, |i| self.ring().smallest_positive_lift(iso.map_back(ring.clone_el(values.at(bitreverse(i, self.log2_n()))))) as u64);
        let output = self.out_of_place_negacyclic_fft::<_, true>(input, &ring);
        for i in 0..(self.len as usize) {
            *values.at_mut(i) = iso.map(self.ring().coerce(&StaticRing::<i64>::RING, output[i] as i64));
        }
    }

    pub fn root_of_unity(&self) -> &El<Zn> {
        &self.root_of_unity
    }
}

#[cfg(test)]
use feanor_math::{default_memory_provider, assert_el_eq};

#[test]
fn test_unordered_ntt() {
    let ring = Zn::new(17);

    // length 4
    {
        let z = ring.from_int(9);
        let inv_z = ring.invert(&z).unwrap();
        let fft = HEXLNTT::new(4, ring, z, default_memory_provider!());
        let one = fft.ring().one();
        let zero = fft.ring().zero();

        let mut values = [zero, one, zero, zero];
        let original = values;

        // bitreversed order
        let expected = [inv_z, ring.pow(inv_z, 5), ring.pow(inv_z, 3), ring.pow(inv_z, 7)];

        fft.unordered_negacyclic_ntt(&mut values, fft.ring(), &default_memory_provider!());
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &expected[i], &values[i]);
        }

        fft.unordered_negacyclic_inv_ntt(&mut values, fft.ring(), &default_memory_provider!());
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &original[i], &values[i]);
        }
    }
    // length 8
    {
        let z = ring.from_int(3);
        let inv_z = ring.invert(&z).unwrap();
        let fft = HEXLNTT::new(8, ring, z, default_memory_provider!());
        let one = fft.ring().one();
        let zero = fft.ring().zero();

        let mut values = [zero, one, one, zero, zero, zero, zero, zero];
        let original = values;

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

        fft.unordered_negacyclic_ntt(&mut values, fft.ring(), &default_memory_provider!());
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &expected[i], &values[i]);
        }

        fft.unordered_negacyclic_inv_ntt(&mut values, fft.ring(), &default_memory_provider!());
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &original[i], &values[i]);
        }
    }
}

#[test]
fn test_ntt() {
    let ring = Zn::new(17);

    // length 4
    {
        let z = ring.from_int(9);
        let inv_z = ring.invert(&z).unwrap();
        let fft = HEXLNTT::new(4, ring, z, default_memory_provider!());
        let one = fft.ring().one();
        let zero = fft.ring().zero();

        let mut values = [zero, one, zero, zero];
        let original = values;

        // bitreversed order
        let expected = [inv_z, ring.pow(inv_z, 3), ring.pow(inv_z, 5), ring.pow(inv_z, 7)];

        fft.negacyclic_ntt(&mut values, fft.ring(), &default_memory_provider!());
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &expected[i], &values[i]);
        }

        fft.negacyclic_inv_ntt(&mut values, fft.ring(), &default_memory_provider!());
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &original[i], &values[i]);
        }
    }
    // length 8
    {
        let z = ring.from_int(3);
        let inv_z = ring.invert(&z).unwrap();
        let fft = HEXLNTT::new(8, ring, z, default_memory_provider!());
        let one = fft.ring().one();
        let zero = fft.ring().zero();

        let mut values = [zero, one, one, zero, zero, zero, zero, zero];
        let original = values;

        // bitreversed order
        let expected = [
            ring.add(inv_z, ring.pow(inv_z, 2)),
            ring.add(ring.pow(inv_z, 3), ring.pow(inv_z, 6)),
            ring.add(ring.pow(inv_z, 5), ring.pow(inv_z, 10)),
            ring.add(ring.pow(inv_z, 7), ring.pow(inv_z, 14)),
            ring.add(ring.pow(inv_z, 9), ring.pow(inv_z, 18)),
            ring.add(ring.pow(inv_z, 11), ring.pow(inv_z, 22)),
            ring.add(ring.pow(inv_z, 13), ring.pow(inv_z, 26)),
            ring.add(ring.pow(inv_z, 15),ring.pow(inv_z, 30))
        ];

        fft.negacyclic_ntt(&mut values, fft.ring(), &default_memory_provider!());
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &expected[i], &values[i]);
        }

        fft.negacyclic_inv_ntt(&mut values, fft.ring(), &default_memory_provider!());
        for i in 0..4 {
            assert_el_eq!(fft.ring(), &original[i], &values[i]);
        }
    }
}
