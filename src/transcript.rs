//! This module contains utilities and traits for dealing with Fiat-Shamir
//! transcripts.

use ff::Field;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::arithmetic::{CurveAffine, FieldExt};

/// This is a generic interface for a sponge function that can be used for
/// Fiat-Shamir transformations.
pub trait Hasher<F: FieldExt>: Clone + Send + Sync + 'static {
    /// Initialize the sponge with some key.
    fn init(key: F) -> Self;
    /// Absorb a field element into the sponge.
    fn absorb(&mut self, value: F);
    /// Square a field element out of the sponge.
    fn squeeze(&mut self) -> F;
}

/// This is just a simple (and completely broken) hash function, standing in for
/// some algebraic hash function that we'll switch to later.
#[derive(Debug, Clone)]
pub struct DummyHash<F: FieldExt> {
    power: F,
    state: F,
}

impl<F: FieldExt> Hasher<F> for DummyHash<F> {
    fn init(key: F) -> Self {
        DummyHash {
            power: F::ZETA + F::one() + key,
            state: F::ZETA,
        }
    }
    fn absorb(&mut self, value: F) {
        for _ in 0..10 {
            self.state += value;
            self.state *= self.power;
            self.power += self.power.square();
            self.state += self.power;
        }
    }
    fn squeeze(&mut self) -> F {
        let tmp = self.state;
        self.absorb(tmp);
        tmp
    }
}

/// A transcript that can absorb points from both the base field and scalar
/// field of a curve
#[derive(Debug, Clone)]
pub struct Transcript<C: CurveAffine, HBase, HScalar>
where
    HBase: Hasher<C::Base>,
    HScalar: Hasher<C::Scalar>,
{
    // Hasher over the base field
    base_hasher: HBase,
    // Hasher over the scalar field
    scalar_hasher: HScalar,
    // Indicates if scalar(s) has been hashed but not squeezed
    scalar_needs_squeezing: bool,
    // PhantomData
    _marker: PhantomData<C>,
}

impl<C: CurveAffine, HBase: Hasher<C::Base>, HScalar: Hasher<C::Scalar>>
    Transcript<C, HBase, HScalar>
{
    /// Initialise a new transcript with Field::one() as keys
    /// in both the base_hasher and scalar_hasher
    pub fn new() -> Self {
        let base_hasher = HBase::init(C::Base::one());
        let scalar_hasher = HScalar::init(C::Scalar::one());
        Transcript {
            base_hasher,
            scalar_hasher,
            scalar_needs_squeezing: false,
            _marker: PhantomData,
        }
    }

    fn conditional_scalar_squeeze(&mut self) {
        if self.scalar_needs_squeezing {
            let transcript_scalar_point =
                C::Base::from_bytes(&(self.scalar_hasher.squeeze()).to_bytes()).unwrap();
            self.base_hasher.absorb(transcript_scalar_point);
            self.scalar_needs_squeezing = false;
        }
    }

    /// Absorb a curve point into the transcript by absorbing
    /// its x and y coordinates
    pub fn absorb_point(&mut self, point: &C) -> Result<(), ()> {
        self.conditional_scalar_squeeze();
        let tmp = point.get_xy();
        if bool::from(tmp.is_none()) {
            return Err(());
        };
        let tmp = tmp.unwrap();
        self.base_hasher.absorb(tmp.0);
        self.base_hasher.absorb(tmp.1);
        Ok(())
    }

    /// Absorb a base into the base_hasher
    pub fn absorb_base(&mut self, base: C::Base) {
        self.conditional_scalar_squeeze();
        self.base_hasher.absorb(base);
    }

    /// Absorb a scalar into the scalar_hasher
    pub fn absorb_scalar(&mut self, scalar: C::Scalar) {
        self.scalar_hasher.absorb(scalar);
        self.scalar_needs_squeezing = true;
    }

    /// Squeeze the transcript to obtain a C::Base value.
    pub fn squeeze(&mut self) -> C::Base {
        self.conditional_scalar_squeeze();
        self.base_hasher.squeeze()
    }
}

/// This is a 128-bit verifier challenge.
#[derive(Copy, Clone, Debug)]
pub struct Challenge(pub(crate) u128);

impl Challenge {
    /// Obtains a new challenge from the transcript.
    pub fn get<C, HBase, HScalar>(transcript: &mut Transcript<C, HBase, HScalar>) -> Challenge
    where
        C: CurveAffine,
        HBase: Hasher<C::Base>,
        HScalar: Hasher<C::Scalar>,
    {
        Challenge(transcript.squeeze().get_lower_128())
    }
}

/// The scalar representation of a verifier challenge.
///
/// The `T` type can be used to scope the challenge to a specific context, or set to `()`
/// if no context is required.
#[derive(Copy, Clone, Debug)]
pub struct ChallengeScalar<F: FieldExt, T> {
    inner: F,
    _marker: PhantomData<T>,
}

impl<F: FieldExt, T> From<Challenge> for ChallengeScalar<F, T> {
    /// This algorithm applies the mapping of Algorithm 1 from the
    /// [Halo](https://eprint.iacr.org/2019/1021) paper.
    fn from(challenge: Challenge) -> Self {
        let mut acc = (F::ZETA + F::one()).double();

        for i in (0..64).rev() {
            let should_negate = ((challenge.0 >> ((i << 1) + 1)) & 1) == 1;
            let should_endo = ((challenge.0 >> (i << 1)) & 1) == 1;

            let q = if should_negate { -F::one() } else { F::one() };
            let q = if should_endo { q * F::ZETA } else { q };
            acc = acc + q + acc;
        }

        ChallengeScalar {
            inner: acc,
            _marker: PhantomData::default(),
        }
    }
}

impl<F: FieldExt, T> ChallengeScalar<F, T> {
    /// Obtains a new challenge from the transcript.
    pub fn get<C, HBase, HScalar>(transcript: &mut Transcript<C, HBase, HScalar>) -> Self
    where
        C: CurveAffine,
        HBase: Hasher<C::Base>,
        HScalar: Hasher<C::Scalar>,
    {
        Challenge::get(transcript).into()
    }
}

impl<F: FieldExt, T> Deref for ChallengeScalar<F, T> {
    type Target = F;

    fn deref(&self) -> &F {
        &self.inner
    }
}
