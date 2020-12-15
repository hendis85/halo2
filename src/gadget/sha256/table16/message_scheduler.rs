use std::convert::TryInto;

use super::{super::BLOCK_SIZE, BlockWord, SpreadInputs, Table16Chip, ROUNDS};
use crate::{
    arithmetic::FieldExt,
    gadget::{Cell, Layouter},
    plonk::{Advice, Column, ConstraintSystem, Error, Fixed},
};

#[derive(Clone, Debug)]
pub(super) struct MessageWord {
    var: Cell,
    value: Option<u32>,
}

#[derive(Clone, Debug)]
pub(super) struct MessageScheduler {
    lookup: SpreadInputs,
    message_schedule: Column<Advice>,
    extras: [Column<Advice>; 4],

    /// Construct a word using reduce_4.
    s_word: Column<Fixed>,
    /// Decomposition gate for W_1, W_63, W_64.
    s_decompose_0: Column<Fixed>,
    /// Decomposition gate for W_{2..14}
    s_decompose_1: Column<Fixed>,
    /// Decomposition gate for W_{15..49}
    s_decompose_2: Column<Fixed>,
    /// Decomposition gate for W_{50..62}
    s_decompose_3: Column<Fixed>,
}

impl MessageScheduler {
    /// Configures the message scheduler.
    ///
    /// `message_schedule` is the column into which the message schedule will be placed.
    /// The caller must create appropriate permutations in order to load schedule words
    /// into the compression rounds.
    ///
    /// `extras` contains columns that the message scheduler will only use for internal
    /// gates, and will not place any constraints on (such as lookup constraints) outside
    /// itself.
    pub(super) fn new<F: FieldExt>(
        meta: &mut ConstraintSystem<F>,
        lookup: SpreadInputs,
        message_schedule: Column<Advice>,
        extras: [Column<Advice>; 4],
    ) -> Self {
        // Create fixed columns for the selectors we will require.
        let s_word = meta.fixed_column();
        let s_decompose_0 = meta.fixed_column();
        let s_decompose_1 = meta.fixed_column();
        let s_decompose_2 = meta.fixed_column();
        let s_decompose_3 = meta.fixed_column();

        // Rename these here for ease of matching the gates to the specification.
        let a_0 = lookup.tag;
        let a_1 = lookup.dense;
        let a_2 = lookup.spread;
        let a_3 = extras[0];
        let a_4 = extras[1];
        let a_5 = message_schedule;
        let a_6 = extras[2];
        let a_7 = extras[3];

        // meta.create_gate(|meta| {});

        MessageScheduler {
            lookup,
            message_schedule,
            extras,
            s_word,
            s_decompose_0,
            s_decompose_1,
            s_decompose_2,
            s_decompose_3,
        }
    }

    pub(super) fn process<F: FieldExt>(
        &self,
        layouter: &mut impl Layouter<Table16Chip<F>>,
        input: [BlockWord; BLOCK_SIZE],
    ) -> Result<[MessageWord; ROUNDS], Error> {
        let mut w = vec![];

        layouter.assign_region(|mut region| {
            // Assign the block words into the message schedule.
            for i in 0..16 {
                let var = region.assign_advice(self.message_schedule, 2 * i, || {
                    input[i]
                        .value
                        .map(|v| F::from_u64(v as u64))
                        .ok_or(Error::SynthesisError)
                })?;

                w.push(MessageWord {
                    var,
                    value: input[i].value,
                });
            }

            Ok(())
        })?;

        Ok(w.try_into().unwrap())
    }
}
