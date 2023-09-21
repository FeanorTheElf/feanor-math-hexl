use cpp::cpp;

use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::rings::zn::zn_42::Zn;
use feanor_math::rings::zn::*;
use feanor_math::integer::IntegerRingStore;
use feanor_math::mempool::{MemoryProvider, AllocatingMemoryProvider};
use feanor_math::algorithms::fft::*;
use feanor_math::ring::*;
use feanor_math::primitive_int::*;
use feanor_math::vector::*;

use std::ffi::c_void;
use std::ops::DerefMut;

cpp!{{
    #include "hexl/ntt/ntt.hpp"

    intel::hexl::NTT* create_ntt(uint64_t N, uint64_t modulus, uint64_t root_of_unity) {
        return new intel::hexl::NTT(N, modulus, root_of_unity);
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

    pub fn new(N: u64, ring: Zn, root_of_unity: El<Zn>, memory_provider: M) -> Self {
        // N must be power of two
        assert!(1 << StaticRing::<i64>::RING.abs_log2_ceil(&(N as i64)).unwrap() == N);
        let q = *ring.modulus() as u64;
        let root_of_unity_scalar = ring.smallest_positive_lift(root_of_unity) as u64;
        let pointer = unsafe { cpp!( [N as "uint64_t", q as "uint64_t", root_of_unity as "uint64_t"] -> *mut c_void as "void*" { return create_ntt(N, q, root_of_unity); } ) };
        HEXLNTT { data: pointer, ring: ring, len: N, memory_provider: memory_provider, root_of_unity: root_of_unity }
    }
}

impl<M: MemoryProvider<u64>> Drop for HEXLNTT<M> {

    fn drop(&mut self) {
        let ptr = self.data;
        unsafe { cpp!( [ptr as "void*"] { destroy_ntt(static_cast<intel::hexl::NTT*>(ptr)); } ) }
    }
}

impl<M: MemoryProvider<u64>> HEXLNTT<M> {


    fn len(&self) -> usize {
        self.len as usize
    }

    fn ring(&self) -> &Zn {
        &self.ring
    }

    fn unordered_negacyclic_fft<V, S, M2>(&self, mut values: V, ring: S, _memory_provider: &M2)
        where S: RingStore, 
            S::Type: CanonicalIso<<Zn as RingStore>::Type>, 
            V: VectorViewMut<feanor_math::ring::El<S>>,
            M2: MemoryProvider<feanor_math::ring::El<S>>
    {
        let iso = ring.can_iso(self.ring()).unwrap();
        let mut input = self.memory_provider.get_new_init(self.len as usize, |i| self.ring().smallest_positive_lift(iso.map_back(ring.clone_el(values.at(i)))) as u64);
        let mut output = self.memory_provider.get_new_init(self.len as usize, |_| 0);
        {
            let ptr = self.data;
            let input_ptr = input.deref_mut().as_ptr();
            let output_ptr = output.deref_mut().as_ptr();
            unsafe { cpp!( [ptr as "void*", input_ptr as "const uint64_t*", output_ptr as "uint64_t*"] { ntt_forward(static_cast<intel::hexl::NTT*>(ptr), input_ptr, output_ptr); } ) }
        }
        for i in 0..(self.len as usize) {
            *values.at_mut(i) = iso.map(self.ring().coerce(&StaticRing::<i64>::RING, output[i] as i64));
        }
    }

    fn unordered_negacyclic_inv_fft<V, S, M2>(&self, mut values: V, ring: S, _memory_provider: &M2)
        where S: RingStore, 
            S::Type: CanonicalIso<<Zn as RingStore>::Type>, 
            V: VectorViewMut<feanor_math::ring::El<S>>,
            M2: MemoryProvider<feanor_math::ring::El<S>>
    {
        let iso = ring.can_iso(self.ring()).unwrap();
        let mut input = self.memory_provider.get_new_init(self.len as usize, |i| self.ring().smallest_positive_lift(iso.map_back(ring.clone_el(values.at(i)))) as u64);
        let mut output = self.memory_provider.get_new_init(self.len as usize, |_| 0);
        {
            let ptr = self.data;
            let input_ptr = input.deref_mut().as_ptr();
            let output_ptr = output.deref_mut().as_ptr();
            unsafe { cpp!( [ptr as "void*", input_ptr as "const uint64_t*", output_ptr as "uint64_t*"] { ntt_backward(static_cast<intel::hexl::NTT*>(ptr), input_ptr, output_ptr); } ) }
        }
        for i in 0..(self.len as usize) {
            *values.at_mut(i) = iso.map(self.ring().coerce(&StaticRing::<i64>::RING, output[i] as i64));
        }
    }

    fn root_of_unity(&self) -> &El<Zn> {
        &self.root_of_unity
    }
}

#[cfg(test)]
use feanor_math::{default_memory_provider, assert_el_eq};

#[test]
fn test_ntt() {
    let ring = Zn::new(17);
    let z = ring.from_int(3);
    let inv_z = ring.invert(&z).unwrap();
    let fft = HEXLNTT::new(4, ring, z, default_memory_provider!());
    let one = fft.ring().one();
    let zero = fft.ring().zero();
    let mut values = [zero, one, zero, zero];
    let expected = [inv_z, ring.pow(inv_z, 3), ring.pow(inv_z, 5), ring.pow(inv_z, 7)];
    fft.ring().println(fft.root_of_unity());

    fft.unordered_negacyclic_fft(&mut values, fft.ring(), &default_memory_provider!());
    for i in 0..4 {
        assert_el_eq!(fft.ring(), &expected[i], &values[i]);
    }
}