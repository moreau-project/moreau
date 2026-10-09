# E8 conic data for the CUDA chordal-decomposition stall

`membership-e8.npz` is the conic data of a 2997×297 program (1101 nonnegative rows, 126 second-order cones, PSD cones of orders 6, 6, 6, 6, 6, 24, 8, 8; no zero cones) used to reproduce the CUDA stall reported in the linked issue. It holds A in CSC form (`A_data`, `A_indices`, `A_indptr`, `A_shape`), `b`, `c`, and the cone sizes (`zero`, `nonneg`, `soc`, `psd`), with PSD rows in Moreau's svec order. The problem is: minimize cᵀx subject to Ax + s = b, s in the cones.

This branch holds data only and is not meant to be merged.
