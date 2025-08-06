use std::alloc::{Allocator, Global};
use std::borrow::Cow;
use std::cmp::{min, max};

use tracing::instrument;

use feanor_math::algorithms::convolution::*;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::integer::*;
use feanor_math::rings::zn::*;
use feanor_math::seq::*;
use feanor_math::rings::zn::zn_64::*;

use super::hexl::*;

///
/// Algorithm that computes convolutions by using [`HEXLNegacyclicNTT`] over a fixed base ring.
/// 
pub struct HEXLConvolution<A = Global>
    where A: Allocator + Clone
{
    ring: Zn,
    max_log2_n: usize,
    fft_algos: Vec<HEXLNegacyclicNTT>,
    allocator: A
}

impl HEXLConvolution {

    ///
    /// Creates a [`HEXLConvolution`] that supports convolutions of output length up
    /// to `2^max_log2_n` over the fixed given ring.
    /// 
    pub fn new(ring: Zn, max_log2_n: usize) -> Self {
        Self::new_with(ring, max_log2_n, Global)
    }
}

impl<A> HEXLConvolution<A>
    where A: Allocator + Clone
{
    ///
    /// Creates a [`HEXLConvolution`] that supports convolutions of output length up
    /// to `2^max_log2_n` over the fixed given ring, using the given allocator for temporary 
    /// memory allocations.
    /// 
    pub fn new_with(ring: Zn, max_log2_n: usize, allocator: A) -> Self {
        assert!(max_log2_n <= ring.integer_ring().get_ring().abs_lowest_set_bit(&ring.integer_ring().sub_ref_fst(ring.modulus(), ring.integer_ring().one())).unwrap());
        Self {
            fft_algos: (0..=max_log2_n).map(|log2_n| HEXLNegacyclicNTT::for_zn(ring.clone(), log2_n).unwrap()).collect(),
            ring: ring,
            allocator: allocator,
            max_log2_n: max_log2_n,
        }
    }

    pub fn max_supported_output_len(&self) -> usize {
        1 << self.max_log2_n
    }

    pub fn ring(&self) -> &Zn {
        &self.ring
    }

    ///
    /// Computes the convolution, assuming that both `lhs` and `rhs` store the negacyclic NTTs
    /// of the same power-of-two length.
    /// 
    #[instrument(skip_all)]
    fn compute_convolution_ntt(&self, lhs: Cow<PreparedConvolutionOperand<A>>, rhs: Cow<PreparedConvolutionOperand<A>>) -> Vec<ZnEl, A> {
        let log2_n = ZZ.abs_log2_ceil(&(lhs.data.len() as i64)).unwrap();
        assert_eq!(lhs.data.len(), 1 << log2_n);
        assert_eq!(rhs.data.len(), 1 << log2_n);

        let (mut lhs, rhs) = match rhs {
            Cow::Borrowed(rhs) => (lhs.into_owned().data, &rhs.data),
            Cow::Owned(rhs) => (rhs.data, &lhs.data)
        };
        for i in 0..(1 << log2_n) {
            self.ring.mul_assign_ref(&mut lhs[i], &rhs[i]);
        }
        return lhs;
    }

    fn get_fft<'a>(&'a self, log2_n: usize) -> &'a HEXLNegacyclicNTT {
        &self.fft_algos[log2_n]
    }
    
    #[instrument(skip_all)]
    fn prepare_convolution_base<V: VectorView<El<Zn>>>(&self, val: V, log2_n: usize) -> PreparedConvolutionOperand<A> {
        let mut input = Vec::with_capacity_in(1 << log2_n, &self.allocator);
        input.extend(val.as_iter().map(|x| self.ring.clone_el(x)));
        input.resize_with(1 << log2_n, || self.ring.zero());
        let mut result = Vec::with_capacity_in(1 << log2_n, self.allocator.clone());
        result.resize_with(1 << log2_n, || self.ring.zero());
        let fft = self.get_fft(log2_n);
        fft.unordered_negacyclic_fft_base::<false>(&mut input[..], &mut result[..]);
        return PreparedConvolutionOperand {
            data: result
        };
    }
}

pub struct PreparedConvolutionOperand<A = Global>
    where A: Allocator + Clone
{
    data: Vec<El<Zn>, A>
}

impl<A> Clone for PreparedConvolutionOperand<A>
    where A: Allocator + Clone
{
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
    }
}

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

impl<A> ConvolutionAlgorithm<ZnBase> for HEXLConvolution<A>
    where A: Allocator + Clone
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<A>;

    fn supports_ring<S: RingStore<Type = ZnBase> + Copy>(&self, ring: S) -> bool {
        ring.get_ring() == self.ring.get_ring()
    }

    #[instrument(skip_all)]
    fn compute_convolution<S: RingStore<Type = ZnBase> + Copy, V1: VectorView<<ZnBase as RingBase>::Element>, V2: VectorView<<ZnBase as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [<ZnBase as RingBase>::Element], ring: S) {
        self.compute_convolution_prepared(lhs, None, rhs, None, dst, ring)
    }

    #[instrument(skip_all)]
    fn prepare_convolution_operand<S, V>(&self, val: V, length_hint: Option<usize>, ring: S) -> Self::PreparedConvolutionOperand
        where S: RingStore<Type = ZnBase> + Copy, V: VectorView<ZnEl>
    {
        assert!(self.supports_ring(&ring));
        let log2_n_in = ZZ.abs_log2_ceil(&(val.len() as i64)).unwrap();
        let log2_n_out = log2_n_in + 1;
        let log2_hint = length_hint.and_then(|l| ZZ.abs_log2_ceil(&(l as i64))).unwrap_or(0);
        return self.prepare_convolution_base(val, max(log2_n_out, log2_hint));
    }

    fn compute_convolution_prepared<S, V1, V2>(&self, lhs: V1, lhs_prep: Option<&Self::PreparedConvolutionOperand>, rhs: V2, rhs_prep: Option<&Self::PreparedConvolutionOperand>, dst: &mut [ZnEl], ring: S)
        where S: RingStore<Type = ZnBase> + Copy, V1: VectorView<ZnEl>, V2: VectorView<ZnEl>
    {
        assert!(self.supports_ring(&ring));
        let log2_n_out = ZZ.abs_log2_ceil(&((lhs.len() + rhs.len()) as i64)).unwrap();

        let lhs_ntt = if lhs_prep.is_some() && lhs_prep.unwrap().data.len() == (1 << log2_n_out) {
            Cow::Borrowed(lhs_prep.unwrap())
        } else {
            Cow::Owned(self.prepare_convolution_base(lhs, log2_n_out))
        };
        let rhs_ntt = if rhs_prep.is_some() && rhs_prep.unwrap().data.len() == (1 << log2_n_out) {
            Cow::Borrowed(rhs_prep.unwrap())
        } else {
            Cow::Owned(self.prepare_convolution_base(rhs, log2_n_out))
        };

        let mut tmp_ntt = self.compute_convolution_ntt(lhs_ntt, rhs_ntt);

        let mut tmp_out = Vec::with_capacity_in(1 << log2_n_out, &self.allocator);
        tmp_out.resize_with(1 << log2_n_out, || self.ring.zero());
        self.get_fft(log2_n_out).unordered_negacyclic_fft_base::<true>(&mut tmp_ntt[..], &mut tmp_out[..]);
        for i in 0..min(dst.len(), 1 << log2_n_out) {
            self.ring.add_assign_ref(&mut dst[i], &tmp_out[i]);
        }
    }
    
    fn compute_convolution_sum<'a, S, I, V1, V2>(&self, values: I, dst: &mut [ZnEl], ring: S) 
        where S: RingStore<Type = ZnBase> + Copy, 
            I: ExactSizeIterator<Item = (V1, Option<&'a Self::PreparedConvolutionOperand>, V2, Option<&'a Self::PreparedConvolutionOperand>)>,
            V1: VectorView<ZnEl>,
            V2: VectorView<ZnEl>,
            Self: 'a
    {
        let len = dst.len() - 1;
        let log2_n_out = StaticRing::<i64>::RING.abs_log2_ceil(&len.try_into().unwrap()).unwrap();

        let mut buffer = Vec::with_capacity_in(1 << log2_n_out, self.allocator.clone());
        buffer.resize_with(1 << log2_n_out, || ring.zero());

        for (lhs, lhs_prep, rhs, rhs_prep) in values {
            assert!(lhs.len() + rhs.len() - 1 <= len);
            if lhs.len() == 0 || rhs.len() == 0 {
                return;
            }

            let lhs_ntt = if lhs_prep.is_some() && lhs_prep.unwrap().data.len() == (1 << log2_n_out) {
                Cow::Borrowed(lhs_prep.unwrap())
            } else {
                Cow::Owned(self.prepare_convolution_base(lhs, log2_n_out))
            };
            let rhs_ntt = if rhs_prep.is_some() && rhs_prep.unwrap().data.len() == (1 << log2_n_out) {
                Cow::Borrowed(rhs_prep.unwrap())
            } else {
                Cow::Owned(self.prepare_convolution_base(rhs, log2_n_out))
            };

            for (i, x) in self.compute_convolution_ntt(lhs_ntt, rhs_ntt).into_iter().enumerate() {
                ring.add_assign(&mut buffer[i], x);
            }
        }
        let mut tmp_out = Vec::with_capacity_in(1 << log2_n_out, &self.allocator);
        tmp_out.resize_with(1 << log2_n_out, || self.ring.zero());
        self.get_fft(log2_n_out).unordered_negacyclic_fft_base::<true>(&mut buffer[..], &mut tmp_out[..]);
        for i in 0..min(dst.len(), 1 << log2_n_out) {
            self.ring.add_assign_ref(&mut dst[i], &tmp_out[i]);
        }
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::homomorphism::Homomorphism;

#[test]
fn test_convolution() {
    let ring = Zn::new(65537);
    let convolutor = HEXLConvolution::new_with(ring, 15, Global);

    let check = |lhs: &[ZnEl], rhs: &[ZnEl], add: &[ZnEl]| {
        let mut expected = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        STANDARD_CONVOLUTION.compute_convolution(lhs, rhs, &mut expected, &ring);

        let mut actual1 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution(lhs, rhs, &mut actual1, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual1[i]);
        }
        
        let lhs_prepared = convolutor.prepare_convolution_operand(lhs, None, &ring);
        let rhs_prepared = convolutor.prepare_convolution_operand(rhs, None, &ring);

        let mut actual2 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_prepared(lhs, Some(&lhs_prepared), rhs, None, &mut actual2, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual2[i]);
        }
        
        let mut actual3 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_prepared(lhs, None, rhs, Some(&rhs_prepared), &mut actual3, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual3[i]);
        }
        
        let mut actual4 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_prepared(lhs, Some(&lhs_prepared), rhs, Some(&rhs_prepared), &mut actual4, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual4[i]);
        }
    };

    for lhs_len in [1, 2, 3, 4, 7, 8, 9] {
        for rhs_len in [1, 5, 8, 16, 17] {
            let lhs = (0..lhs_len).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
            let rhs = (0..rhs_len).map(|i| ring.int_hom().map(16 * i)).collect::<Vec<_>>();
            let add = (0..(lhs_len + rhs_len)).map(|i| ring.int_hom().map(32768 * i)).collect::<Vec<_>>();
            check(&lhs, &rhs, &add);
        }
    }
}

#[test]
fn test_convolution_generic() {
    let ring = Zn::new(65537);
    let convolutor = HEXLConvolution::new_with(ring, 15, Global);
    feanor_math::algorithms::convolution::generic_tests::test_convolution(convolutor, ring, ring.one());
}

#[test]
fn test_convolution_inner_product() {
    let ring = Zn::new(65537);
    let convolutor = HEXLConvolution::new_with(ring, 5, Global);

    for l1_len in [7, 8, 13] {
        for l2_len in [8, 9] {
            for r1_len  in [5, 7] {
                for r2_len in [4, 15] {
                    let l1 = (0..l1_len).map(|i| ring.int_hom().map(i)).collect::<Vec<_>>();
                    let l2 = (0..l2_len).map(|i| ring.int_hom().map(4 * i)).collect::<Vec<_>>();
                    let r1 = (0..r1_len).map(|i| ring.int_hom().map(16 * i)).collect::<Vec<_>>();
                    let r2 = (0..r2_len).map(|i| ring.int_hom().map(64 * i)).collect::<Vec<_>>();
                    let mut expected = (0..32).map(|_| ring.zero()).collect::<Vec<_>>();
                    STANDARD_CONVOLUTION.compute_convolution(&l1, &r1, &mut expected, ring);
                    STANDARD_CONVOLUTION.compute_convolution(&l2, &r2, &mut expected, ring);
                    let mut actual = (0..32).map(|_| ring.zero()).collect::<Vec<_>>();
                    convolutor.compute_convolution_sum([
                            (&l1, Some(&convolutor.prepare_convolution_operand(&l1, None, ring)), &r1, Some(&convolutor.prepare_convolution_operand(&r1, None, ring))), 
                            (&l2, Some(&convolutor.prepare_convolution_operand(&l2, None, ring)), &r2, None)
                        ].into_iter(), 
                        &mut actual, 
                        ring
                    );
                    for i in 0..32 {
                        assert_el_eq!(ring, expected[i], actual[i]);
                    }
                }
            }
        }
    }
}