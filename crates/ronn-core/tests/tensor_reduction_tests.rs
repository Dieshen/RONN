//! Comprehensive tests for tensor reduction operations.
//!
//! This module tests all reduction operations including:
//! - Sum, mean, max, min
//! - Variance, standard deviation
//! - Norms (L1, L2, Lp)
//! - Softmax, argmax, argmin
//! - Cumulative operations

mod test_utils;

use anyhow::Result;
use ronn_core::{DataType, ReductionOps, Tensor, TensorLayout};
use test_utils::*;

#[test]
fn test_sum_all() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let sum = a.sum_all()?;
    assert_tensor_eq(&sum, &[21.0])?; // 1+2+3+4+5+6 = 21
    Ok(())
}

#[test]
fn test_sum_along_dimension_0() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let sum = a.sum_dim(0, false)?;
    assert_tensor_eq(&sum, &[5.0, 7.0, 9.0])?; // [1+4, 2+5, 3+6]
    Ok(())
}

#[test]
fn test_sum_along_dimension_1() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let sum = a.sum_dim(1, false)?;
    assert_tensor_eq(&sum, &[6.0, 15.0])?; // [1+2+3, 4+5+6]
    Ok(())
}

#[test]
fn test_sum_keep_dim() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let sum_keep = a.sum_dim(1, true)?;
    assert_eq!(sum_keep.shape(), vec![2, 1]);

    let sum_no_keep = a.sum_dim(1, false)?;
    assert_eq!(sum_no_keep.shape(), vec![2]);
    Ok(())
}

#[test]
fn test_mean_all() -> Result<()> {
    let a = Tensor::from_data(
        vec![2.0, 4.0, 6.0, 8.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let mean = a.mean_all()?;
    assert_tensor_eq(&mean, &[5.0])?; // (2+4+6+8)/4 = 5
    Ok(())
}

#[test]
fn test_mean_along_dimension() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let mean = a.mean_dim(0, false)?;
    assert_tensor_eq(&mean, &[2.0, 3.0])?; // [(1+3)/2, (2+4)/2]
    Ok(())
}

#[test]
fn test_max_all() -> Result<()> {
    let a = Tensor::from_data(
        vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
        vec![2, 4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let max = a.max_all()?;
    assert_tensor_eq(&max, &[9.0])?;
    Ok(())
}

#[test]
fn test_max_along_dimension() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 5.0, 3.0, 2.0, 4.0, 1.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let max = a.max_dim(1, false)?;
    let max_data = max.to_vec()?;
    // Max of each row: [5.0, 4.0]
    assert!((max_data[0] - 5.0).abs() < 1e-6);
    assert!((max_data[1] - 4.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_min_all() -> Result<()> {
    let a = Tensor::from_data(
        vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let min = a.min_all()?;
    assert_tensor_eq(&min, &[1.0])?;
    Ok(())
}

#[test]
fn test_min_along_dimension() -> Result<()> {
    let a = Tensor::from_data(
        vec![5.0, 2.0, 8.0, 1.0, 3.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let min = a.min_dim(1, false)?;
    let min_data = min.to_vec()?;
    // Min of each row: [2.0, 1.0]
    assert!((min_data[0] - 2.0).abs() < 1e-6);
    assert!((min_data[1] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_variance() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let var = a.var_all()?;
    let var_data = var.to_vec()?;
    // Variance of [1,2,3,4,5] = 2.0
    assert!((var_data[0] - 2.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_standard_deviation() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let std = a.std_all()?;
    let std_data = std.to_vec()?;
    // Std dev = sqrt(2) ≈ 1.414
    assert!((std_data[0] - 1.4142135).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_l1_norm() -> Result<()> {
    let a = Tensor::from_data(
        vec![-3.0, 4.0, -5.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let norm = a.norm_p(1.0)?;
    assert_tensor_eq(&norm, &[12.0])?; // |−3| + |4| + |−5| = 12
    Ok(())
}

#[test]
fn test_l2_norm() -> Result<()> {
    let a = Tensor::from_data(
        vec![3.0, 4.0],
        vec![2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let norm = a.norm()?;
    assert_tensor_eq(&norm, &[5.0])?; // sqrt(9 + 16) = 5
    Ok(())
}

#[test]
fn test_lp_norms() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // L1 norm
    let l1 = a.norm_p(1.0)?;
    assert_tensor_eq(&l1, &[6.0])?;

    // L2 norm
    let l2 = a.norm_p(2.0)?;
    let l2_data = l2.to_vec()?;
    assert!((l2_data[0] - (14.0_f32).sqrt()).abs() < 1e-5);

    // L∞ norm (max absolute value)
    let linf = a.norm_p(f32::INFINITY)?;
    assert_tensor_eq(&linf, &[3.0])?;

    Ok(())
}

#[test]
fn test_invalid_norm_p() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    // Negative p should fail
    assert!(a.norm_p(-1.0).is_err());

    // Zero p should fail
    assert!(a.norm_p(0.0).is_err());
}

#[test]
fn test_softmax() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let softmax = a.softmax(0)?;
    let softmax_data = softmax.to_vec()?;

    // Sum should be 1
    let sum: f32 = softmax_data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // All values should be positive
    assert!(softmax_data.iter().all(|&x| x > 0.0));

    // Values should be in ascending order for ascending inputs
    assert!(softmax_data[0] < softmax_data[1]);
    assert!(softmax_data[1] < softmax_data[2]);
    Ok(())
}

#[test]
fn test_softmax_2d() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let softmax = a.softmax(1)?;
    assert_eq!(softmax.shape(), vec![2, 3]);

    let data = softmax.to_vec()?;

    // Each row should sum to 1
    let row1_sum = data[0] + data[1] + data[2];
    let row2_sum = data[3] + data[4] + data[5];
    assert!((row1_sum - 1.0).abs() < 1e-5);
    assert!((row2_sum - 1.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_log_softmax() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let log_softmax = a.log_softmax(0)?;
    let data = log_softmax.to_vec()?;

    // All values should be negative (since log of probability < 1)
    assert!(data.iter().all(|&x| x < 0.0));
    Ok(())
}

#[test]
fn test_argmax() -> Result<()> {
    let a = Tensor::from_data(
        vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let argmax = a.argmax(1, false)?;
    assert_eq!(argmax.dtype(), DataType::U32);
    assert_eq!(argmax.shape(), vec![2]);
    Ok(())
}

#[test]
fn test_argmin() -> Result<()> {
    let a = Tensor::from_data(
        vec![3.0, 1.0, 4.0, 5.0, 2.0, 9.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let argmin = a.argmin(1, false)?;
    assert_eq!(argmin.dtype(), DataType::U32);
    assert_eq!(argmin.shape(), vec![2]);
    Ok(())
}

#[test]
fn test_argmax_keep_dim() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let argmax_keep = a.argmax(1, true)?;
    assert_eq!(argmax_keep.shape(), vec![2, 1]);

    let argmax_no_keep = a.argmax(1, false)?;
    assert_eq!(argmax_no_keep.shape(), vec![2]);
    Ok(())
}

#[test]
fn test_prod_all() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let prod = a.prod_all()?;
    assert_tensor_eq(&prod, &[24.0])?; // 1*2*3*4 = 24
    Ok(())
}

#[test]
fn test_cumsum() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let cumsum = a.cumsum(0)?;
    assert_tensor_eq(&cumsum, &[1.0, 3.0, 6.0, 10.0])?;
    Ok(())
}

#[test]
fn test_cumprod() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let cumprod = a.cumprod(0)?;
    assert_tensor_eq(&cumprod, &[1.0, 2.0, 6.0, 24.0])?;
    Ok(())
}

#[test]
fn test_count_nonzero() -> Result<()> {
    let a = Tensor::from_data(
        vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
        vec![6],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let count = a.count_nonzero()?;
    assert_eq!(count, 3);
    Ok(())
}

#[test]
fn test_reduction_on_empty_dimensions() -> Result<()> {
    let a = Tensor::from_data(vec![1.0], vec![1], DataType::F32, TensorLayout::RowMajor)?;

    let sum = a.sum_all()?;
    assert_tensor_eq(&sum, &[1.0])?;

    let mean = a.mean_all()?;
    assert_tensor_eq(&mean, &[1.0])?;
    Ok(())
}

#[test]
fn test_multi_dimensional_reductions() -> Result<()> {
    let a = create_sequential_tensor(vec![2, 3, 4], DataType::F32)?;

    let sum_all = a.sum_all()?;
    assert_eq!(sum_all.numel(), 1);

    let sum_dim0 = a.sum_dim(0, false)?;
    assert_eq!(sum_dim0.shape(), vec![3, 4]);

    let sum_dim1 = a.sum_dim(1, false)?;
    assert_eq!(sum_dim1.shape(), vec![2, 4]);

    let sum_dim2 = a.sum_dim(2, false)?;
    assert_eq!(sum_dim2.shape(), vec![2, 3]);
    Ok(())
}

#[test]
fn test_reduction_errors() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    // Out of bounds dimension
    assert!(a.sum_dim(5, false).is_err());
    assert!(a.max_dim(10, false).is_err());
    assert!(a.argmax(3, false).is_err());
}

#[test]
fn test_numerical_stability_softmax() -> Result<()> {
    // Large values should not cause overflow
    let a = Tensor::from_data(
        vec![1000.0, 1001.0, 1002.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let softmax = a.softmax(0)?;
    let data = softmax.to_vec()?;

    // Should still sum to 1
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Should not have NaN or Inf
    assert!(data.iter().all(|&x| x.is_finite()));
    Ok(())
}

#[test]
fn test_reduction_preserves_dtype() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let sum = a.sum_all()?;
    assert_eq!(sum.dtype(), DataType::F32);

    let mean = a.mean_all()?;
    assert_eq!(mean.dtype(), DataType::F32);
    Ok(())
}
