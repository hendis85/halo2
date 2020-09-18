//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use crate::arithmetic::CurveAffine;
use crate::poly::{
    commitment::OpeningProof, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
    Polynomial,
};
use crate::transcript::Hasher;

mod circuit;
mod prover;
mod srs;
mod verifier;

pub use circuit::*;
pub use prover::*;
pub use srs::*;
pub use verifier::*;

/// This is a structured reference string (SRS) that is (deterministically)
/// computed from a specific circuit and parameters for the polynomial
/// commitment scheme.
#[derive(Debug)]
pub struct SRS<C: CurveAffine> {
    domain: EvaluationDomain<C::Scalar>,
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    fixed_commitments: Vec<C>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    fixed_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    permutation_commitments: Vec<Vec<C>>,
    permutations: Vec<Vec<Polynomial<C::Scalar, LagrangeCoeff>>>,
    permutation_polys: Vec<Vec<Polynomial<C::Scalar, Coeff>>>,
    permutation_cosets: Vec<Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>>,
    cs: ConstraintSystem<C::Scalar>,
}

/// This is an object which represents a (Turbo)PLONK proof.
// This structure must never allow points at infinity.
#[derive(Debug, Clone)]
pub struct Proof<C: CurveAffine> {
    advice_commitments: Vec<C>,
    h_commitments: Vec<C>,
    permutation_product_commitments: Vec<C>,
    permutation_product_evals: Vec<C::Scalar>,
    permutation_product_inv_evals: Vec<C::Scalar>,
    permutation_evals: Vec<Vec<C::Scalar>>,
    advice_evals: Vec<C::Scalar>,
    aux_evals: Vec<C::Scalar>,
    fixed_evals: Vec<C::Scalar>,
    h_evals: Vec<C::Scalar>,
    f_commitment: C,
    q_evals: Vec<C::Scalar>,
    opening: OpeningProof<C>,
}

/// This is an error that could occur during proving or circuit synthesis.
// TODO: these errors need to be cleaned up
#[derive(Debug)]
pub enum Error {
    /// This is an error that can occur during synthesis of the circuit, for
    /// example, when the witness is not present.
    SynthesisError,
    /// The structured reference string or the parameters are not compatible
    /// with the circuit being synthesized.
    IncompatibleParams,
    /// The constraint system is not satisfied.
    ConstraintSystemFailure,
    /// Out of bounds index passed to a backend
    BoundsFailure,
    /// Opening error
    OpeningError,
}

fn hash_point<C: CurveAffine, H: Hasher<C::Base>>(
    transcript: &mut H,
    point: &C,
) -> Result<(), Error> {
    let tmp = point.get_xy();
    if bool::from(tmp.is_none()) {
        return Err(Error::SynthesisError);
    };
    let tmp = tmp.unwrap();
    transcript.absorb(tmp.0);
    transcript.absorb(tmp.1);
    Ok(())
}

#[test]
fn test_proving() {
    use crate::arithmetic::{Curve, EqAffine, Field, Fp, Fq};
    use crate::poly::commitment::{Blind, Params};
    use crate::transcript::DummyHash;
    use std::marker::PhantomData;
    const K: u32 = 5;

    /// This represents an advice wire at a certain row in the ConstraintSystem
    #[derive(Copy, Clone, Debug)]
    pub struct Variable(AdviceWire, usize);

    /// This represents an auxiliary wire at a certain row in the ConstraintSystem
    #[derive(Copy, Clone, Debug)]
    pub struct AuxVariable(AuxWire, usize);

    /// This represents a wire at a certain row in the ConstraintSystem
    #[derive(Copy, Clone, Debug)]
    pub struct PermVariable(Wire, usize);

    // Initialize the polynomial commitment parameters
    let params: Params<EqAffine> = Params::new::<DummyHash<Fq>>(K);

    struct PLONKConfig {
        a: AdviceWire,
        b: AdviceWire,
        c: AdviceWire,
        d: AdviceWire,
        e: AdviceWire,

        x: AuxWire,

        sa: FixedWire,
        sb: FixedWire,
        sc: FixedWire,
        sm: FixedWire,
        sx: FixedWire,

        perm: usize,
        perm2: usize,
    }

    trait StandardCS<FF: Field> {
        fn raw_multiply<F>(&mut self, f: F) -> Result<(Variable, Variable, Variable), Error>
        where
            F: FnOnce() -> Result<(FF, FF, FF), Error>;
        fn raw_add<F>(&mut self, f: F) -> Result<(Variable, Variable, Variable), Error>
        where
            F: FnOnce() -> Result<(FF, FF, FF), Error>;
        fn copy(&mut self, a: PermVariable, b: PermVariable) -> Result<(), Error>;
        fn raw_aux<F>(&mut self, f: F) -> Result<(Variable, AuxVariable), Error>
        where
            F: FnOnce() -> Result<(FF, FF), Error>;
    }

    struct MyCircuit<F: Field> {
        a: Option<F>,
        x: Option<F>,
    }

    struct StandardPLONK<'a, F: Field, CS: Assignment<F> + 'a> {
        cs: &'a mut CS,
        config: PLONKConfig,
        current_gate: usize,
        _marker: PhantomData<F>,
    }

    impl<'a, FF: Field, CS: Assignment<FF>> StandardPLONK<'a, FF, CS> {
        fn new(cs: &'a mut CS, config: PLONKConfig) -> Self {
            StandardPLONK {
                cs,
                config,
                current_gate: 0,
                _marker: PhantomData,
            }
        }
    }

    impl<'a, FF: Field, CS: Assignment<FF>> StandardCS<FF> for StandardPLONK<'a, FF, CS> {
        fn raw_multiply<F>(&mut self, f: F) -> Result<(Variable, Variable, Variable), Error>
        where
            F: FnOnce() -> Result<(FF, FF, FF), Error>,
        {
            let index = self.current_gate;
            self.current_gate += 1;
            let mut value = None;
            self.cs.assign_advice(self.config.a, index, || {
                value = Some(f()?);
                Ok(value.ok_or(Error::SynthesisError)?.0)
            })?;
            self.cs.assign_advice(self.config.d, index, || {
                Ok(value.ok_or(Error::SynthesisError)?.0.square().square())
            })?;
            self.cs.assign_advice(self.config.b, index, || {
                Ok(value.ok_or(Error::SynthesisError)?.1)
            })?;
            self.cs.assign_advice(self.config.e, index, || {
                Ok(value.ok_or(Error::SynthesisError)?.1.square().square())
            })?;
            self.cs.assign_advice(self.config.c, index, || {
                Ok(value.ok_or(Error::SynthesisError)?.2)
            })?;

            self.cs
                .assign_fixed(self.config.sa, index, || Ok(FF::zero()))?;
            self.cs
                .assign_fixed(self.config.sb, index, || Ok(FF::zero()))?;
            self.cs
                .assign_fixed(self.config.sc, index, || Ok(FF::one()))?;
            self.cs
                .assign_fixed(self.config.sm, index, || Ok(FF::one()))?;
            Ok((
                Variable(self.config.a, index),
                Variable(self.config.b, index),
                Variable(self.config.c, index),
            ))
        }
        fn raw_add<F>(&mut self, f: F) -> Result<(Variable, Variable, Variable), Error>
        where
            F: FnOnce() -> Result<(FF, FF, FF), Error>,
        {
            let index = self.current_gate;
            self.current_gate += 1;
            let mut value = None;
            self.cs.assign_advice(self.config.a, index, || {
                value = Some(f()?);
                Ok(value.ok_or(Error::SynthesisError)?.0)
            })?;
            self.cs.assign_advice(self.config.d, index, || {
                Ok(value.ok_or(Error::SynthesisError)?.0.square().square())
            })?;
            self.cs.assign_advice(self.config.b, index, || {
                Ok(value.ok_or(Error::SynthesisError)?.1)
            })?;
            self.cs.assign_advice(self.config.e, index, || {
                Ok(value.ok_or(Error::SynthesisError)?.1.square().square())
            })?;
            self.cs.assign_advice(self.config.c, index, || {
                Ok(value.ok_or(Error::SynthesisError)?.2)
            })?;

            self.cs
                .assign_fixed(self.config.sa, index, || Ok(FF::one()))?;
            self.cs
                .assign_fixed(self.config.sb, index, || Ok(FF::one()))?;
            self.cs
                .assign_fixed(self.config.sc, index, || Ok(FF::one()))?;
            self.cs
                .assign_fixed(self.config.sm, index, || Ok(FF::zero()))?;
            Ok((
                Variable(self.config.a, index),
                Variable(self.config.b, index),
                Variable(self.config.c, index),
            ))
        }
        fn copy(&mut self, left: PermVariable, right: PermVariable) -> Result<(), Error> {
            let left_wire = match left.0 {
                Wire::Advice(wire) => match wire {
                    x if x == self.config.a => 0,
                    x if x == self.config.b => 1,
                    x if x == self.config.c => 2,
                    _ => unreachable!(),
                },
                Wire::Aux(wire) => match wire {
                    x if x == self.config.x => 3,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            let right_wire = match right.0 {
                Wire::Advice(wire) => match wire {
                    x if x == self.config.a => 0,
                    x if x == self.config.b => 1,
                    x if x == self.config.c => 2,
                    _ => unreachable!(),
                },
                Wire::Aux(wire) => match wire {
                    x if x == self.config.x => 3,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            self.cs
                .copy(self.config.perm, left_wire, left.1, right_wire, right.1)?;
            self.cs
                .copy(self.config.perm2, left_wire, left.1, right_wire, right.1)
        }
        fn raw_aux<F>(&mut self, f: F) -> Result<(Variable, AuxVariable), Error>
        where
            F: FnOnce() -> Result<(FF, FF), Error>,
        {
            let index = self.current_gate;
            self.current_gate += 1;
            let mut value = None;
            self.cs.assign_advice(self.config.a, index, || {
                value = Some(f()?);
                Ok(value.ok_or(Error::SynthesisError)?.0)
            })?;
            self.cs
                .assign_fixed(self.config.sx, index, || Ok(FF::zero()))?;
            Ok((
                Variable(self.config.a, index),
                AuxVariable(self.config.x, index),
            ))
        }
    }

    impl<F: Field> Circuit<F> for MyCircuit<F> {
        type Config = PLONKConfig;

        fn configure(meta: &mut ConstraintSystem<F>) -> PLONKConfig {
            let e = meta.advice_wire();
            let a = meta.advice_wire();
            let b = meta.advice_wire();
            let sf = meta.fixed_wire();
            let c = meta.advice_wire();
            let d = meta.advice_wire();

            let x = meta.aux_wire();

            let perm = meta.permutation(&[
                Wire::Advice(a),
                Wire::Advice(b),
                Wire::Advice(c),
                Wire::Aux(x),
            ]);
            let perm2 = meta.permutation(&[
                Wire::Advice(a),
                Wire::Advice(b),
                Wire::Advice(c),
                Wire::Aux(x),
            ]);

            let sm = meta.fixed_wire();
            let sa = meta.fixed_wire();
            let sb = meta.fixed_wire();
            let sc = meta.fixed_wire();
            let sx = meta.fixed_wire();

            meta.create_gate(|meta| {
                let d = meta.query_advice(d, 1);
                let a = meta.query_advice(a, 0);
                let sf = meta.query_fixed(sf, 0);
                let e = meta.query_advice(e, -1);
                let b = meta.query_advice(b, 0);
                let c = meta.query_advice(c, 0);

                let x = meta.query_aux(x, 0);

                let sa = meta.query_fixed(sa, 0);
                let sb = meta.query_fixed(sb, 0);
                let sc = meta.query_fixed(sc, 0);
                let sm = meta.query_fixed(sm, 0);
                let sx = meta.query_fixed(sx, 0);

                a.clone() * sa
                    + b.clone() * sb
                    + a * b * sm
                    + (c * sc * (-F::one()))
                    + sf * (d * e)
                    + (x * sx * (-F::one()))
            });

            PLONKConfig {
                a,
                b,
                c,
                d,
                e,
                x,
                sa,
                sb,
                sc,
                sm,
                sx,
                perm,
                perm2,
            }
        }

        fn synthesize(
            &self,
            cs: &mut impl Assignment<F>,
            config: PLONKConfig,
        ) -> Result<(), Error> {
            let mut cs = StandardPLONK::new(cs, config);

            for _ in 0..10 {
                let mut a_squared = None;
                let (a0, _, c0) = cs.raw_multiply(|| {
                    a_squared = self.a.map(|a| a.square());
                    Ok((
                        self.a.ok_or(Error::SynthesisError)?,
                        self.a.ok_or(Error::SynthesisError)?,
                        a_squared.ok_or(Error::SynthesisError)?,
                    ))
                })?;
                let (a1, b1, _) = cs.raw_add(|| {
                    let fin = a_squared.and_then(|a2| self.a.map(|a| a + a2));
                    Ok((
                        self.a.ok_or(Error::SynthesisError)?,
                        a_squared.ok_or(Error::SynthesisError)?,
                        fin.ok_or(Error::SynthesisError)?,
                    ))
                })?;
                cs.copy(
                    PermVariable(Wire::Advice(a0.0), a0.1),
                    PermVariable(Wire::Advice(a1.0), a1.1),
                )?;
                cs.copy(
                    PermVariable(Wire::Advice(b1.0), b1.1),
                    PermVariable(Wire::Advice(c0.0), c0.1),
                )?;
            }
            let (_, x) = cs.raw_aux(|| {
                Ok((
                    self.x.ok_or(Error::SynthesisError)?,
                    self.x.ok_or(Error::SynthesisError)?,
                ))
            })?;
            cs.copy(
                PermVariable(Wire::Aux(x.0), x.1),
                PermVariable(Wire::Aux(x.0), x.1),
            )?;

            Ok(())
        }
    }

    let empty_circuit: MyCircuit<Fp> = MyCircuit { a: None, x: None };

    // Initialize the SRS
    let srs = SRS::generate(&params, &empty_circuit).expect("SRS generation should not fail");

    // TODO: use meaningful value from recursion
    let mut aux_lagrange_polys = vec![srs.domain.empty_lagrange(); srs.cs.num_aux_wires];

    // TODO: use meaningful value from recursion
    let mut aux_commitments: Vec<EqAffine> = vec![];
    for poly in &aux_lagrange_polys {
        let commitment = params.commit_lagrange(poly, Blind::default());
        aux_commitments.push(commitment.to_affine());
    }

    for _ in 0..100 {
        // Generate circuit
        let circuit: MyCircuit<Fp> = MyCircuit {
            a: Some(Fp::random()),

            // TODO: use meaningful value from recursion
            x: Some(Fp::random()),
        };

        // Create a proof
        let proof = Proof::create::<DummyHash<Fq>, DummyHash<Fp>, _>(
            &params,
            &srs,
            &circuit,
            aux_lagrange_polys.clone(),
        )
        .expect("proof generation should not fail");

        let msm = params.empty_msm();
        let guard = proof
            .verify::<DummyHash<Fq>, DummyHash<Fp>>(&params, &srs, msm, aux_commitments.clone())
            .unwrap();
        {
            let msm = guard.clone().use_challenges();
            assert!(msm.is_zero());
        }
        {
            let g = guard.compute_g();
            let (msm, _) = guard.clone().use_g(g);
            assert!(msm.is_zero());
        }
        let msm = guard.clone().use_challenges();
        assert!(msm.clone().is_zero());
        let guard = proof
            .verify::<DummyHash<Fq>, DummyHash<Fp>>(&params, &srs, msm, aux_commitments.clone())
            .unwrap();
        {
            let msm = guard.clone().use_challenges();
            assert!(msm.is_zero());
        }
        {
            let g = guard.compute_g();
            let (msm, _) = guard.clone().use_g(g);
            assert!(msm.clone().is_zero());

            let mut g_scalars = vec![Fp::one(); 1 << K];
            if let Some(msm_g_scalars) = msm.get_g_scalars() {
                g_scalars = msm_g_scalars;
            }
            let g_lagrange_poly = srs.domain.lagrange_from_vec(g_scalars.clone());
            aux_lagrange_polys = vec![g_lagrange_poly.clone(); 1];
            let g_commitment = params
                .commit_lagrange(&g_lagrange_poly, Blind::default())
                .to_affine();
            aux_commitments = vec![g_commitment; 1];
        }
    }
}
