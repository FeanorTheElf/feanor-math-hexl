use std::alloc::{Allocator, Global};
use std::cmp::min;

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

    #[instrument(skip_all)]
    fn add_assign_elementwise_product(lhs: &[ZnEl], rhs: &[ZnEl], dst: &mut [ZnEl], ring: &Zn) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), dst.len());
        for i in 0..lhs.len() {
            ring.add_assign(&mut dst[i], ring.mul_ref(&lhs[i], &rhs[i]));
        }
    }

    ///
    /// Computes the convolution, assuming that both `lhs` and `rhs` store the negacyclic NTTs
    /// of the same power-of-two length.
    /// 
    #[instrument(skip_all)]
    fn compute_convolution_base(&self, mut lhs: PreparedConvolutionOperand<A>, rhs: &PreparedConvolutionOperand<A>, out: &mut [El<Zn>]) {
        let log2_n = ZZ.abs_log2_ceil(&(lhs.data.len() as i64)).unwrap();
        assert_eq!(lhs.data.len(), 1 << log2_n);
        assert_eq!(rhs.data.len(), 1 << log2_n);
        assert!(lhs.len + rhs.len <= 1 << log2_n);
        assert!(out.len() >= lhs.len + rhs.len);
        for i in 0..(1 << log2_n) {
            self.ring.mul_assign_ref(&mut lhs.data[i], &rhs.data[i]);
        }
        let mut tmp = Vec::with_capacity_in(1 << log2_n, &self.allocator);
        tmp.resize_with(1 << log2_n, || self.ring.zero());
        self.get_fft(log2_n).unordered_negacyclic_fft_base::<true>(&mut lhs.data[..], &mut tmp[..]);
        for i in 0..min(out.len(), 1 << log2_n) {
            self.ring.add_assign_ref(&mut out[i], &tmp[i]);
        }
    }

    #[instrument(skip_all)]
    fn un_and_redo_fft(&self, input: &[ZnEl], log2_n: usize) -> Vec<ZnEl, A> {
        let log2_in_len = ZZ.abs_log2_ceil(&(input.len() as i64)).unwrap();
        assert_eq!(input.len(), 1 << log2_in_len);
        assert!(log2_in_len < log2_n);

        let mut tmp1 = Vec::with_capacity_in(input.len(), self.allocator.clone());
        tmp1.extend(input.iter().map(|x| self.ring.clone_el(x)));
        let mut tmp2 = Vec::with_capacity_in(input.len(), &self.allocator);
        tmp2.resize_with(input.len(), || self.ring.zero());
        self.get_fft(log2_in_len).unordered_negacyclic_fft_base::<true>(&mut tmp1[..], &mut tmp2[..]);

        tmp1.resize_with(1 << log2_n, || self.ring.zero());
        tmp2.resize_with(1 << log2_n, || self.ring.zero());
        self.get_fft(log2_n).unordered_negacyclic_fft_base::<false>(&mut tmp2[..], &mut tmp1[..]);
        return tmp1;
    }

    fn get_fft<'a>(&'a self, log2_n: usize) -> &'a HEXLNegacyclicNTT {
        &self.fft_algos[log2_n]
    }

    fn clone_prepared_operand(&self, operand: &PreparedConvolutionOperand<A>) -> PreparedConvolutionOperand<A> {
        let mut result = Vec::with_capacity_in(operand.data.len(), self.allocator.clone());
        result.extend(operand.data.iter().map(|x| self.ring.clone_el(x)));
        return PreparedConvolutionOperand {
            len: operand.len,
            data: result
        };
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
            len: val.len(),
            data: result
        };
    }
}

impl<A> ConvolutionAlgorithm<ZnBase> for HEXLConvolution<A>
    where A: Allocator + Clone
{
    fn supports_ring<S: RingStore<Type = ZnBase> + Copy>(&self, ring: S) -> bool {
        ring.get_ring() == self.ring.get_ring()
    }

    #[instrument(skip_all)]
    fn compute_convolution<S: RingStore<Type = ZnBase> + Copy, V1: VectorView<<ZnBase as RingBase>::Element>, V2: VectorView<<ZnBase as RingBase>::Element>>(&self, lhs: V1, rhs: V2, dst: &mut [<ZnBase as RingBase>::Element], ring: S) {
        assert!(self.supports_ring(&ring));
        let log2_n = ZZ.abs_log2_ceil(&((lhs.len() + rhs.len()) as i64)).unwrap();
        let lhs_prep = self.prepare_convolution_base(lhs, log2_n);
        let rhs_prep = self.prepare_convolution_base(rhs, log2_n);
        self.compute_convolution_base(lhs_prep, &rhs_prep, dst);
    }
}

pub struct PreparedConvolutionOperand<A = Global>
    where A: Allocator + Clone
{
    len: usize,
    data: Vec<El<Zn>, A>
}

const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

impl<A> PreparedConvolutionAlgorithm<ZnBase> for HEXLConvolution<A>
    where A: Allocator + Clone,
        A: 'static
{
    type PreparedConvolutionOperand = PreparedConvolutionOperand<A>;

    #[instrument(skip_all)]
    fn prepare_convolution_operand<S: RingStore<Type = ZnBase> + Copy, V: VectorView<El<Zn>>>(&self, val: V, ring: S) -> Self::PreparedConvolutionOperand {
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_n_in = ZZ.abs_log2_ceil(&(val.len() as i64)).unwrap();
        let log2_n_out = log2_n_in + 1;
        return self.prepare_convolution_base(val, log2_n_out);
    }

    #[instrument(skip_all)]
    fn compute_convolution_lhs_prepared<S: RingStore<Type = ZnBase> + Copy, V: VectorView<El<Zn>>>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: V, dst: &mut [El<Zn>], ring: S) {
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_lhs = ZZ.abs_log2_ceil(&(lhs.data.len() as i64)).unwrap();
        assert_eq!(lhs.data.len(), 1 << log2_lhs);
        let log2_n = ZZ.abs_log2_ceil(&((lhs.len + rhs.len()) as i64)).unwrap().max(log2_lhs);
        assert!(log2_lhs <= log2_n);
        self.compute_convolution_prepared(lhs, &self.prepare_convolution_base(rhs, log2_n), dst, ring);
    }

    #[instrument(skip_all)]
    fn compute_convolution_prepared<S: RingStore<Type = ZnBase> + Copy>(&self, lhs: &Self::PreparedConvolutionOperand, rhs: &Self::PreparedConvolutionOperand, dst: &mut [El<Zn>], ring: S) {
        assert!(ring.get_ring() == self.ring.get_ring());
        let log2_lhs = ZZ.abs_log2_ceil(&(lhs.data.len() as i64)).unwrap();
        assert_eq!(1 << log2_lhs, lhs.data.len());
        let log2_rhs = ZZ.abs_log2_ceil(&(rhs.data.len() as i64)).unwrap();
        assert_eq!(1 << log2_rhs, rhs.data.len());
        match log2_lhs.cmp(&log2_rhs) {
            std::cmp::Ordering::Equal => self.compute_convolution_base(self.clone_prepared_operand(lhs), rhs, dst),
            std::cmp::Ordering::Greater => self.compute_convolution_base(PreparedConvolutionOperand { data: self.un_and_redo_fft(&rhs.data, log2_lhs), len: rhs.len }, lhs, dst),
            std::cmp::Ordering::Less => self.compute_convolution_base(PreparedConvolutionOperand { data: self.un_and_redo_fft(&lhs.data, log2_rhs), len: lhs.len }, rhs, dst)
        }
    }

    #[instrument(skip_all)]
    fn compute_convolution_inner_product_prepared<'a, S, I>(&self, values: I, dst: &mut [ZnEl], ring: S)
        where S: RingStore<Type = ZnBase> + Copy, 
            I: Iterator<Item = (&'a Self::PreparedConvolutionOperand, &'a Self::PreparedConvolutionOperand)>,
            PreparedConvolutionOperand<A>: 'a
    {
        assert!(ring.get_ring() == self.ring.get_ring());
        let mut values_it = values.peekable();
        if values_it.peek().is_none() {
            return;
        }
        let expected_len = values_it.peek().unwrap().0.data.len().max(values_it.peek().unwrap().1.data.len());
        let mut current_log2_len = ZZ.abs_log2_ceil(&(expected_len as i64)).unwrap();
        assert_eq!(expected_len, 1 << current_log2_len);
        let mut tmp1 = Vec::with_capacity_in(1 << current_log2_len, self.allocator.clone());
        tmp1.resize_with(1 << current_log2_len, || ring.zero());
        for (lhs, rhs) in values_it {
            assert!(dst.len() >= lhs.len + rhs.len);
            let lhs_log2_len = ZZ.abs_log2_ceil(&(lhs.data.len() as i64)).unwrap();
            let rhs_log2_len = ZZ.abs_log2_ceil(&(rhs.data.len() as i64)).unwrap();
            let new_log2_len = current_log2_len.max(lhs_log2_len).max(rhs_log2_len);
            
            if current_log2_len < new_log2_len {
                tmp1 = self.un_and_redo_fft(&tmp1, new_log2_len);
                current_log2_len = new_log2_len;
            }
            match (lhs_log2_len < current_log2_len, rhs_log2_len < current_log2_len) {
                (false, false) => Self::add_assign_elementwise_product(&lhs.data, &rhs.data, &mut tmp1, RingValue::from_ref(ring.get_ring())),
                (true, false) => Self::add_assign_elementwise_product(&self.un_and_redo_fft(&lhs.data, new_log2_len), &rhs.data, &mut tmp1, RingValue::from_ref(ring.get_ring())),
                (false, true) => Self::add_assign_elementwise_product(&lhs.data, &self.un_and_redo_fft(&rhs.data, new_log2_len), &mut tmp1, RingValue::from_ref(ring.get_ring())),
                (true, true) => Self::add_assign_elementwise_product(&self.un_and_redo_fft(&lhs.data, new_log2_len), &self.un_and_redo_fft(&rhs.data, new_log2_len), &mut tmp1, RingValue::from_ref(ring.get_ring())),
            }
        }
        let mut tmp2 = Vec::with_capacity_in(1 << current_log2_len, &self.allocator);
        tmp2.resize_with(1 << current_log2_len, || self.ring.zero());
        self.get_fft(current_log2_len).unordered_negacyclic_fft_base::<true>(&mut tmp1, &mut tmp2);
        for i in 0..min(dst.len(), 1 << current_log2_len) {
            self.ring.add_assign_ref(&mut dst[i], &tmp2[i]);
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
        
        let lhs_prepared = convolutor.prepare_convolution_operand(lhs, &ring);
        let rhs_prepared = convolutor.prepare_convolution_operand(rhs, &ring);

        let mut actual2 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_lhs_prepared(&lhs_prepared, rhs, &mut actual2, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual2[i]);
        }
        
        let mut actual3 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_rhs_prepared(lhs, &rhs_prepared, &mut actual3, &ring);
        for i in 0..(lhs.len() + rhs.len()) {
            assert_el_eq!(&ring, &expected[i], &actual3[i]);
        }
        
        let mut actual4 = (0..(lhs.len() + rhs.len())).map(|i| if i < add.len() { add[i] } else { ring.zero() }).collect::<Vec<_>>();
        convolutor.compute_convolution_prepared(&lhs_prepared, &rhs_prepared, &mut actual4, &ring);
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
                    convolutor.compute_convolution_inner_product_prepared([
                            (&convolutor.prepare_convolution_operand(l1, ring), &convolutor.prepare_convolution_operand(r1, ring)), 
                            (&convolutor.prepare_convolution_operand(l2, ring), &convolutor.prepare_convolution_operand(r2, ring))
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