#![allow(clippy::too_many_arguments)]

// -----------------------------------
// reverse the compact decomposition
// -----------------------------------

use crate::algebra::*;
use crate::solver::chordal::ChordalInfo;
use crate::solver::chordal::SparsityPattern;
use crate::solver::core::cones::*;
use crate::solver::DefaultVariables;
use std::cmp::max;
use std::iter::zip;
use std::ops::Range;

impl<T> ChordalInfo<T>
where
    T: FloatT,
{
    pub(crate) fn decomp_reverse_compact(
        &self,
        new_vars: &mut DefaultVariables<T>,
        old_vars: &DefaultVariables<T>,
        old_cones: &[SupportedConeT<T>],
    ) {
        let old_s = &old_vars.s;
        let old_z = &old_vars.z;
        let new_s = &mut new_vars.s;
        let new_z = &mut new_vars.z;

        // the cones for the originating problem, i.e. the cones
        // that are compatible with the new_vars we want to populate,
        // are held in chordal_info.init_cones

        let cone_maps = self.cone_maps.as_ref().unwrap();
        let row_ranges: Vec<Range<usize>> = self.init_cones.rng_cones_iter().collect();
        let mut row_ptr = 0;

        // add the blocks for each cone requires a buffer in which
        // to hold sorted cliques.   Allocate it here to avoid
        // repeated allocations.  The size of each clique is
        // never bigger than the associated nblk
        let mut clique_buffer = vec![0; self.largest_nblk()];

        for (cone, cone_map) in zip(old_cones.iter(), cone_maps.iter()) {
            let row_range = row_ranges[cone_map.orig_index].clone();

            if cone_map.tree_and_clique.is_none() {
                row_ptr =
                    add_blocks_with_cone(new_s, old_s, new_z, old_z, row_range, cone, row_ptr);
            } else {
                assert!(matches!(cone, SupportedConeT::PSDTriangleConeT(_)));

                let (tree_index, clique_index) = cone_map.tree_and_clique.unwrap();
                let pattern = &self.spatterns[tree_index];

                row_ptr = add_blocks_with_sparsity_pattern(
                    new_s,
                    old_s,
                    new_z,
                    old_z,
                    row_range,
                    pattern,
                    clique_index,
                    &mut clique_buffer,
                    row_ptr,
                );
            }
        }
    }

    /// Adjoint of `decomp_reverse_compact`: maps upstream gradients (ds_orig, dz_orig)
    /// from original dimensions to augmented (decomposed) dimensions.
    ///
    /// Forward: new_s[orig] += old_s[aug]  (accumulation → adjoint is scatter/broadcast)
    ///          new_z[orig]  = old_z[aug]  (overwrite → adjoint is last-writer only)
    ///
    /// Returns (ds_aug, dz_aug) in augmented dimensions.
    pub(crate) fn adjoint_decomp_reverse_compact(
        &self,
        ds_orig: &[T],
        dz_orig: &[T],
        old_cones: &[SupportedConeT<T>],
        m_aug: usize,
    ) -> (Vec<T>, Vec<T>) {
        let cone_maps = self.cone_maps.as_ref().unwrap();
        let row_ranges: Vec<Range<usize>> = self.init_cones.rng_cones_iter().collect();

        let mut ds_aug = vec![T::zero(); m_aug];
        let mut dz_aug = vec![T::zero(); m_aug];

        // Track the last aug_row that writes to each orig_row (for dz adjoint)
        let mut last_aug_for_orig: Vec<Option<usize>> = vec![None; ds_orig.len()];

        let mut clique_buffer = vec![0; self.largest_nblk()];
        let mut row_ptr = 0;

        // First pass: compute row_ptr positions and last-writer tracking for dz
        for (cone, cone_map) in zip(old_cones.iter(), cone_maps.iter()) {
            let row_range = row_ranges[cone_map.orig_index].clone();

            if cone_map.tree_and_clique.is_none() {
                // Non-decomposed: 1:1 mapping
                let nvars = cone.nvars();
                for k in 0..nvars {
                    let orig_row = row_range.start + k;
                    let aug_row = row_ptr + k;
                    last_aug_for_orig[orig_row] = Some(aug_row);
                }
                row_ptr += nvars;
            } else {
                let (tree_index, clique_index) = cone_map.tree_and_clique.unwrap();
                let pattern = &self.spatterns[tree_index];
                let sntree = &pattern.sntree;
                let ordering = &pattern.ordering;

                let clique = sntree.get_clique(clique_index);
                clique_buffer.resize(clique.len(), 0);
                for (i, &v) in clique.iter().enumerate() {
                    clique_buffer[i] = ordering[v];
                }
                clique_buffer.sort();

                let mut counter = 0;
                for &j in clique_buffer.iter() {
                    for &i in clique_buffer.iter() {
                        if i <= j {
                            let offset = coord_to_upper_triangular_index((i, j));
                            let orig_row = row_range.start + offset;
                            let aug_row = row_ptr + counter;
                            last_aug_for_orig[orig_row] = Some(aug_row);
                            counter += 1;
                        }
                    }
                }
                row_ptr += triangular_number(clique.len());
            }
        }

        // Second pass: fill ds_aug and dz_aug
        // Reset row_ptr for second pass
        let mut row_ptr = 0;
        for (cone, cone_map) in zip(old_cones.iter(), cone_maps.iter()) {
            let row_range = row_ranges[cone_map.orig_index].clone();

            if cone_map.tree_and_clique.is_none() {
                let nvars = cone.nvars();
                for k in 0..nvars {
                    let orig_row = row_range.start + k;
                    let aug_row = row_ptr + k;
                    // ds: adjoint of accumulation = broadcast
                    ds_aug[aug_row] = ds_orig[orig_row];
                    // dz: only last writer
                    if last_aug_for_orig[orig_row] == Some(aug_row) {
                        dz_aug[aug_row] = dz_orig[orig_row];
                    }
                }
                row_ptr += nvars;
            } else {
                let (tree_index, clique_index) = cone_map.tree_and_clique.unwrap();
                let pattern = &self.spatterns[tree_index];
                let sntree = &pattern.sntree;
                let ordering = &pattern.ordering;

                let clique = sntree.get_clique(clique_index);
                clique_buffer.resize(clique.len(), 0);
                for (i, &v) in clique.iter().enumerate() {
                    clique_buffer[i] = ordering[v];
                }
                clique_buffer.sort();

                let mut counter = 0;
                for &j in clique_buffer.iter() {
                    for &i in clique_buffer.iter() {
                        if i <= j {
                            let offset = coord_to_upper_triangular_index((i, j));
                            let orig_row = row_range.start + offset;
                            let aug_row = row_ptr + counter;
                            // ds: adjoint of accumulation = broadcast
                            ds_aug[aug_row] = ds_orig[orig_row];
                            // dz: only last writer
                            if last_aug_for_orig[orig_row] == Some(aug_row) {
                                dz_aug[aug_row] = dz_orig[orig_row];
                            }
                            counter += 1;
                        }
                    }
                }
                row_ptr += triangular_number(clique.len());
            }
        }

        (ds_aug, dz_aug)
    }

    // the largest nblk across all spatterns
    fn largest_nblk(&self) -> usize {
        let mut max_block = 0;
        for sp in self.spatterns.iter() {
            max_block = max(
                max_block,
                *sp.sntree.nblk.as_ref().unwrap().iter().max().unwrap_or(&0),
            );
        }
        max_block
    }
}

fn add_blocks_with_sparsity_pattern<T>(
    new_s: &mut [T],
    old_s: &[T],
    new_z: &mut [T],
    old_z: &[T],
    row_range: Range<usize>,
    spattern: &SparsityPattern,
    clique_index: usize,
    clique_buffer: &mut Vec<usize>,
    row_ptr: usize,
) -> usize
where
    T: FloatT,
{
    let sntree = &spattern.sntree;
    let ordering = &spattern.ordering;

    // load the clique into the buffer provided
    let clique = sntree.get_clique(clique_index);
    clique_buffer.resize(clique.len(), 0);
    for (i, &v) in clique.iter().enumerate() {
        clique_buffer[i] = ordering[v];
    }
    clique_buffer.sort();

    let mut counter = 0;
    for &j in clique_buffer.iter() {
        for &i in clique_buffer.iter() {
            if i <= j {
                let offset = coord_to_upper_triangular_index((i, j));
                new_s[row_range.start + offset] += old_s[row_ptr + counter];
                // notice: z overwrites (instead of adding) the overlapping entries
                new_z[row_range.start + offset] = old_z[row_ptr + counter];
                counter += 1
            }
        }
    }

    row_ptr + triangular_number(clique.len())
}

fn add_blocks_with_cone<T>(
    new_s: &mut [T],
    old_s: &[T],
    new_z: &mut [T],
    old_z: &[T],
    row_range: Range<usize>,
    cone: &SupportedConeT<T>,
    row_ptr: usize,
) -> usize
where
    T: FloatT,
{
    let src_range = row_ptr..(row_ptr + cone.nvars());
    new_s[row_range.clone()].copy_from(&old_s[src_range.clone()]);
    new_z[row_range].copy_from(&old_z[src_range]);
    row_ptr + cone.nvars()
}
