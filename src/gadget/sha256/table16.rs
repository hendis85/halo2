use std::marker::PhantomData;

use super::{Chip, Layouter, Sha256Instructions};
use crate::{
    arithmetic::FieldExt,
    gadget::ChipConfig,
    plonk::{Advice, Column, ConstraintSystem, Error, Fixed},
};

mod message_scheduler;
use message_scheduler::*;

const BITS_7: usize = 1 << 7;
const BITS_10: usize = 1 << 10;
const BITS_11: usize = 1 << 11;
const BITS_13: usize = 1 << 13;
const BITS_14: usize = 1 << 14;

const ROUNDS: usize = 64;
const STATE: usize = 8;

#[allow(clippy::unreadable_literal)]
const ROUND_CONSTANTS: [u32; ROUNDS] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const IV: [u32; STATE] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// A variable stored in advice columns corresponding to a row of [`SpreadTable`].
#[derive(Clone, Debug)]
struct SpreadVar {
    tag: u8,
    dense_var: (),
    dense_value: Option<u16>,
    spread_var: (),
    spread_value: Option<u32>,
}

#[derive(Clone, Debug)]
struct SpreadInputs {
    tag: Column<Advice>,
    dense: Column<Advice>,
    spread: Column<Advice>,
}

#[derive(Clone, Debug)]
struct SpreadTable {
    table_tag: Column<Fixed>,
    table_dense: Column<Fixed>,
    table_spread: Column<Fixed>,
}

impl SpreadTable {
    fn configure<F: FieldExt>(
        meta: &mut ConstraintSystem<F>,
        tag: Column<Advice>,
        dense: Column<Advice>,
        spread: Column<Advice>,
    ) -> (SpreadInputs, Self) {
        let table_tag = meta.fixed_column();
        let table_dense = meta.fixed_column();
        let table_spread = meta.fixed_column();

        meta.lookup(
            &[tag.into(), dense.into(), spread.into()],
            &[table_tag.into(), table_dense.into(), table_spread.into()],
        );

        (
            SpreadInputs { tag, dense, spread },
            SpreadTable {
                table_tag,
                table_dense,
                table_spread,
            },
        )
    }

    fn generate<F: FieldExt>() -> impl Iterator<Item = (F, F, F)> {
        (1..=(1 << 16)).scan(
            (F::zero(), F::zero(), F::zero()),
            |(tag, dense, spread), i| {
                // We computed this table row in the previous iteration.
                let res = (*tag, *dense, *spread);

                // i holds the zero-indexed row number for the next table row.
                match i {
                    BITS_7 | BITS_10 | BITS_11 | BITS_13 | BITS_14 => *tag += F::one(),
                    _ => (),
                }
                *dense += F::one();
                if i & 1 == 0 {
                    // On even-numbered rows we left-shift by 2 bits.
                    *spread *= F::from_u64(4);
                } else {
                    // On odd-numbered rows we add one.
                    *spread += F::one();
                }

                Some(res)
            },
        )
    }

    fn load<F: FieldExt>(&self, layouter: &mut impl Layouter<Table16Chip<F>>) -> Result<(), Error> {
        layouter.assign_region(|mut gate| {
            // We generate the row values lazily (we only need them during keygen).
            let mut rows = Self::generate::<F>();

            for index in 0..(1 << 16) {
                let mut row = None;
                gate.assign_fixed(self.table_tag, index, || {
                    row = rows.next();
                    row.map(|(tag, _, _)| tag).ok_or(Error::SynthesisError)
                })?;
                gate.assign_fixed(self.table_dense, index, || {
                    row.map(|(_, dense, _)| dense).ok_or(Error::SynthesisError)
                })?;
                gate.assign_fixed(self.table_spread, index, || {
                    row.map(|(_, _, spread)| spread)
                        .ok_or(Error::SynthesisError)
                })?;
            }
            Ok(())
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlockWord {
    var: (),
    value: Option<u32>,
}

/// A variable that represents the `[A,B,C,D]` words of the SHA-256 internal state.
///
/// The structure of this variable is influenced by the following factors:
/// - In `Σ_0(A)` we need `A` to be split into pieces `(a,b,c,d)` of lengths `(2,11,9,10)`
///   bits respectively (counting from the little end), as well as their spread forms.
/// - `Maj(A,B,C)` requires having the bits of each input in spread form. For `A` we can
///   reuse the pieces from `Σ_0(A)`. Since `B` and `C` are assigned from `A` and `B`
///   respectively in each round, we therefore also have the same pieces in earlier rows.
///   We align the columns to make it efficient to copy-constrain these forms where they
///   are needed.
#[derive(Clone, Debug)]
struct AbcdVar {
    chunk_0: SpreadVar,
    chunk_1: SpreadVar,
    chunk_2: SpreadVar,
    chunk_3: SpreadVar,
}

/// A variable that represents the `[E,F,G,H]` words of the SHA-256 internal state.
///
/// The structure of this variable is influenced by the following factors:
/// - In `Σ_1(E)` we need `E` to be split into pieces `(a,b,c,d)` of lengths `(6,5,14,7)`
///   bits respectively (counting from the little end), as well as their spread forms.
/// - `Ch(E,F,G)` requires having the bits of each input in spread form. For `E` we can
///   reuse the pieces from `Σ_1(E)`. Since `F` and `G` are assigned from `E` and `F`
///   respectively in each round, we therefore also have the same pieces in earlier rows.
///   We align the columns to make it efficient to copy-constrain these forms where they
///   are needed.
#[derive(Clone, Debug)]
struct EfghVar {}

/// The internal state for SHA-256.
#[derive(Clone, Debug)]
pub struct State {
    h_0: AbcdVar,
    h_1: AbcdVar,
    h_2: AbcdVar,
    h_3: AbcdVar,
    h_4: EfghVar,
    h_5: EfghVar,
    h_6: EfghVar,
    h_7: EfghVar,
}

#[derive(Clone, Debug)]
struct HPrime {}

/// Configuration for a [`Table16Chip`].
#[derive(Clone, Debug)]
pub struct Table16Config {
    lookup_table: SpreadTable,
    message_scheduler: MessageScheduler,
}

impl ChipConfig for Table16Config {}

/// A chip that implements SHA-256 with a maximum lookup table size of $2^16$.
#[derive(Clone, Debug)]
pub struct Table16Chip<F: FieldExt> {
    _marker: PhantomData<F>,
}

impl<F: FieldExt> Table16Chip<F> {
    /// Configures this chip for use in a circuit.
    ///
    /// TODO: Figure out what the necessary shared columns are.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> Table16Config {
        // Columns required by this chip:
        // - Three advice columns to interact with the lookup table.
        let tag = meta.advice_column();
        let dense = meta.advice_column();
        let spread = meta.advice_column();

        let message_schedule = meta.advice_column();
        let extras = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // - N selector columns.
        let s_ch = meta.fixed_column();
        let s_maj = meta.fixed_column();
        let s_upper_sigma_0 = meta.fixed_column();
        let s_upper_sigma_1 = meta.fixed_column();
        let s_lower_sigma_0 = meta.fixed_column();
        let s_lower_sigma_1 = meta.fixed_column();
        let s_23 = meta.fixed_column();
        let s_33 = meta.fixed_column();

        let (lookup_inputs, lookup_table) = SpreadTable::configure(meta, tag, dense, spread);

        let message_scheduler =
            MessageScheduler::new(meta, lookup_inputs, message_schedule, extras);

        Table16Config {
            lookup_table,
            message_scheduler,
        }
    }
}

impl<F: FieldExt> Chip for Table16Chip<F> {
    type Field = F;
    type Config = Table16Config;

    fn load(layouter: &mut impl Layouter<Self>) -> Result<(), Error> {
        let table = layouter.config().lookup_table.clone();
        table.load(layouter)
    }
}

impl<F: FieldExt> Sha256Instructions for Table16Chip<F> {
    type State = State;
    type BlockWord = BlockWord;

    fn initialization_vector(layouter: &mut impl Layouter<Self>) -> Result<State, Error> {
        todo!()
    }

    fn compress(
        layouter: &mut impl Layouter<Self>,
        initial_state: &Self::State,
        input: [Self::BlockWord; super::BLOCK_SIZE],
    ) -> Result<Self::State, Error> {
        let config = layouter.config().clone();

        let w = config.message_scheduler.process(layouter, input)?;

        todo!()
    }

    fn digest(
        layouter: &mut impl Layouter<Self>,
        state: &Self::State,
    ) -> Result<[Self::BlockWord; super::DIGEST_SIZE], Error> {
        // Copy the dense forms of the state variable chunks down to this gate.
        // Reconstruct the 32-bit dense words.
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;
    use std::fmt;
    use std::marker::PhantomData;

    use super::{MessageScheduler, SpreadInputs, SpreadTable, Table16Chip, Table16Config};
    use crate::{
        arithmetic::FieldExt,
        dev::MockProver,
        gadget::{Cell, DynRegion, Layouter, Permutation, Region},
        pasta::{EqAffine, Fp, Fq},
        plonk::{Advice, Assignment, Circuit, Column, ConstraintSystem, Error, Fixed},
        poly::commitment::Params,
        transcript::DummyHash,
    };

    #[test]
    fn lookup_table() {
        const K: u32 = 16;

        /// This represents an advice column at a certain row in the ConstraintSystem
        #[derive(Copy, Clone, Debug)]
        pub struct Variable(Column<Advice>, usize);

        // Initialize the polynomial commitment parameters
        let params: Params<EqAffine> = Params::new::<DummyHash<Fq>>(K);

        #[derive(Debug)]
        struct MyConfig {
            lookup_inputs: SpreadInputs,
            sha256: Table16Config,
        }

        struct MyCircuit {}

        struct MyLayouter<'a, F: FieldExt, CS: Assignment<F> + 'a> {
            cs: &'a mut CS,
            config: MyConfig,
            regions: Vec<usize>,
            current_gate: usize,
            _marker: PhantomData<F>,
        }

        impl<'a, F: FieldExt, CS: Assignment<F> + 'a> fmt::Debug for MyLayouter<'a, F, CS> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("MyLayouter")
                    .field("config", &self.config)
                    .field("regions", &self.regions)
                    .field("current_gate", &self.current_gate)
                    .finish()
            }
        }

        impl<'a, FF: FieldExt, CS: Assignment<FF>> MyLayouter<'a, FF, CS> {
            fn new(cs: &'a mut CS, config: MyConfig) -> Result<Self, Error> {
                let mut res = MyLayouter {
                    cs,
                    config,
                    regions: vec![],
                    current_gate: 0,
                    _marker: PhantomData,
                };

                let table = res.config.sha256.lookup_table.clone();
                table.load(&mut res)?;

                Ok(res)
            }
        }

        impl<'a, F: FieldExt, CS: Assignment<F> + 'a> Layouter<Table16Chip<F>> for MyLayouter<'a, F, CS> {
            fn config(&self) -> &Table16Config {
                &self.config.sha256
            }

            fn assign_region(
                &mut self,
                assignment: impl FnOnce(Region<'_, Table16Chip<F>>) -> Result<(), Error>,
            ) -> Result<(), Error> {
                let region_index = self.regions.len();
                self.regions.push(self.current_gate);

                let mut region = MyRegion::new(self, region_index);
                assignment(Region {
                    region: &mut region,
                })?;
                self.current_gate += region.row_count;

                Ok(())
            }
        }

        struct MyRegion<'r, 'a, F: FieldExt, CS: Assignment<F> + 'a> {
            layouter: &'r mut MyLayouter<'a, F, CS>,
            region_index: usize,
            row_count: usize,
            _marker: PhantomData<F>,
        }

        impl<'r, 'a, F: FieldExt, CS: Assignment<F> + 'a> fmt::Debug for MyRegion<'r, 'a, F, CS> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("MyRegion")
                    .field("layouter", &self.layouter)
                    .field("region_index", &self.region_index)
                    .field("row_count", &self.row_count)
                    .finish()
            }
        }

        impl<'r, 'a, F: FieldExt, CS: Assignment<F> + 'a> MyRegion<'r, 'a, F, CS> {
            fn new(layouter: &'r mut MyLayouter<'a, F, CS>, region_index: usize) -> Self {
                MyRegion {
                    layouter,
                    region_index,
                    row_count: 0,
                    _marker: PhantomData::default(),
                }
            }
        }

        impl<'r, 'a, F: FieldExt, CS: Assignment<F> + 'a> DynRegion<Table16Chip<F>>
            for MyRegion<'r, 'a, F, CS>
        {
            fn assign_advice<'v>(
                &'v mut self,
                column: Column<Advice>,
                offset: usize,
                to: &'v mut (dyn FnMut() -> Result<F, Error> + 'v),
            ) -> Result<Cell, Error> {
                self.layouter.cs.assign_advice(
                    column,
                    self.layouter.regions[self.region_index] + offset,
                    to,
                )?;
                self.row_count = cmp::max(self.row_count, offset);

                Ok(Cell {
                    region_index: self.region_index,
                    row_offset: offset,
                    column: column.into(),
                })
            }

            fn assign_fixed<'v>(
                &'v mut self,
                column: Column<Fixed>,
                offset: usize,
                to: &'v mut (dyn FnMut() -> Result<F, Error> + 'v),
            ) -> Result<Cell, Error> {
                self.layouter.cs.assign_fixed(
                    column,
                    self.layouter.regions[self.region_index] + offset,
                    to,
                )?;
                self.row_count = cmp::max(self.row_count, offset);
                Ok(Cell {
                    region_index: self.region_index,
                    row_offset: offset,
                    column: column.into(),
                })
            }

            fn constrain_equal(
                &mut self,
                permutation: &Permutation,
                left: Cell,
                right: Cell,
            ) -> Result<(), Error> {
                let left_column = permutation
                    .mapping
                    .iter()
                    .position(|c| c == &left.column)
                    .ok_or(Error::SynthesisError)?;
                let right_column = permutation
                    .mapping
                    .iter()
                    .position(|c| c == &right.column)
                    .ok_or(Error::SynthesisError)?;

                self.layouter.cs.copy(
                    permutation.index,
                    left_column,
                    self.layouter.regions[left.region_index] + left.row_offset,
                    right_column,
                    self.layouter.regions[right.region_index] + right.row_offset,
                )?;

                Ok(())
            }
        }

        impl<F: FieldExt> Circuit<F> for MyCircuit {
            type Config = MyConfig;

            fn configure(meta: &mut ConstraintSystem<F>) -> MyConfig {
                let a = meta.advice_column();
                let b = meta.advice_column();
                let c = meta.advice_column();

                let (lookup_inputs, lookup_table) = SpreadTable::configure(meta, a, b, c);

                let message_schedule = meta.advice_column();
                let extras = [
                    meta.advice_column(),
                    meta.advice_column(),
                    meta.advice_column(),
                    meta.advice_column(),
                ];

                let message_scheduler =
                    MessageScheduler::new(meta, lookup_inputs.clone(), message_schedule, extras);

                MyConfig {
                    lookup_inputs,
                    sha256: Table16Config {
                        lookup_table,
                        message_scheduler,
                    },
                }
            }

            fn synthesize(
                &self,
                cs: &mut impl Assignment<F>,
                config: MyConfig,
            ) -> Result<(), Error> {
                let lookup = config.lookup_inputs.clone();
                let mut layouter = MyLayouter::new(cs, config)?;

                layouter.assign_region(|mut gate| {
                    // Test the zero and one value lookup.
                    gate.assign_advice(lookup.tag, 0, || Ok(F::zero()))?;
                    gate.assign_advice(lookup.dense, 0, || Ok(F::zero()))?;
                    gate.assign_advice(lookup.spread, 0, || Ok(F::zero()))?;

                    gate.assign_advice(lookup.tag, 1, || Ok(F::one()))?;
                    gate.assign_advice(lookup.dense, 1, || Ok(F::one()))?;
                    gate.assign_advice(lookup.spread, 1, || Ok(F::one()))?;

                    // TODO: test random lookup values

                    Ok(())
                })
            }
        }

        let circuit: MyCircuit = MyCircuit {};

        let prover = match MockProver::<Fp>::run(16, &circuit, vec![]) {
            Ok(prover) => prover,
            Err(e) => panic!("{:?}", e),
        };
        assert_eq!(prover.verify(), Ok(()));
    }
}
