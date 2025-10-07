//! Comprehensive tests for tensor arithmetic operations.
//!
//! This module tests all arithmetic operations including:
//! - Element-wise operations: add, sub, mul, div
//! - Scalar operations
//! - Broadcasting
//! - Activation functions
//! - Edge cases and error handling

mod test_utils;

use anyhow::Result;
use ronn_core::{ArithmeticOps, DataType, Tensor, TensorLayout};
use test_utils::*;

#[test]
fn test_element_wise_add() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![5.0, 6.0, 7.0, 8.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.add(&b)?;
    assert_tensor_eq(&result, &[6.0, 8.0, 10.0, 12.0])?;
    Ok(())
}

#[test]
fn test_element_wise_sub() -> Result<()> {
    let a = Tensor::from_data(
        vec![10.0, 20.0, 30.0, 40.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.sub(&b)?;
    assert_tensor_eq(&result, &[9.0, 18.0, 27.0, 36.0])?;
    Ok(())
}

#[test]
fn test_element_wise_mul() -> Result<()> {
    let a = Tensor::from_data(
        vec![2.0, 3.0, 4.0, 5.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![10.0, 10.0, 10.0, 10.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.mul(&b)?;
    assert_tensor_eq(&result, &[20.0, 30.0, 40.0, 50.0])?;
    Ok(())
}

#[test]
fn test_element_wise_div() -> Result<()> {
    let a = Tensor::from_data(
        vec![10.0, 20.0, 30.0, 40.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![2.0, 4.0, 5.0, 8.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.div(&b)?;
    assert_tensor_eq(&result, &[5.0, 5.0, 6.0, 5.0])?;
    Ok(())
}

#[test]
fn test_scalar_add() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.add_scalar(10.0)?;
    assert_tensor_eq(&result, &[11.0, 12.0, 13.0, 14.0])?;
    Ok(())
}

#[test]
fn test_scalar_sub() -> Result<()> {
    let a = Tensor::from_data(
        vec![10.0, 20.0, 30.0, 40.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.sub_scalar(5.0)?;
    assert_tensor_eq(&result, &[5.0, 15.0, 25.0, 35.0])?;
    Ok(())
}

#[test]
fn test_scalar_mul() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.mul_scalar(3.0)?;
    assert_tensor_eq(&result, &[3.0, 6.0, 9.0, 12.0])?;
    Ok(())
}

#[test]
fn test_scalar_div() -> Result<()> {
    let a = Tensor::from_data(
        vec![10.0, 20.0, 30.0, 40.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.div_scalar(10.0)?;
    assert_tensor_eq(&result, &[1.0, 2.0, 3.0, 4.0])?;
    Ok(())
}

#[test]
fn test_broadcasting_1d_to_2d() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![10.0, 20.0, 30.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.add(&b)?;
    assert_tensor_eq(&result, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0])?;
    Ok(())
}

#[test]
fn test_broadcasting_column_vector() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(vec![100.0], vec![1], DataType::F32, TensorLayout::RowMajor)?;

    let result = a.add(&b)?;
    assert_tensor_eq(&result, &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0])?;
    Ok(())
}

#[test]
fn test_broadcasting_shape_compatibility() -> Result<()> {
    let a = Tensor::zeros(vec![3, 1], DataType::F32, TensorLayout::RowMajor)?;
    let b = Tensor::zeros(vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;

    assert!(a.is_broadcastable_with(&b));
    let result = Tensor::broadcast_shape(&[3, 1], &[1, 4])?;
    assert_eq!(result, vec![3, 4]);
    Ok(())
}

#[test]
fn test_broadcasting_incompatible_shapes() {
    let a = Tensor::zeros(vec![3, 2], DataType::F32, TensorLayout::RowMajor).unwrap();
    let b = Tensor::zeros(vec![2, 3], DataType::F32, TensorLayout::RowMajor).unwrap();

    assert!(!a.is_broadcastable_with(&b));
    assert!(a.add(&b).is_err());
}

#[test]
fn test_negation() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, -2.0, 3.0, -4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.neg()?;
    assert_tensor_eq(&result, &[-1.0, 2.0, -3.0, 4.0])?;
    Ok(())
}

#[test]
fn test_absolute_value() -> Result<()> {
    let a = Tensor::from_data(
        vec![-5.0, -3.0, 0.0, 2.0, 4.0],
        vec![5],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.abs()?;
    assert_tensor_eq(&result, &[5.0, 3.0, 0.0, 2.0, 4.0])?;
    Ok(())
}

#[test]
fn test_power() -> Result<()> {
    let a = Tensor::from_data(
        vec![2.0, 3.0, 4.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.pow(2.0)?;
    assert_tensor_eq(&result, &[4.0, 9.0, 16.0])?;
    Ok(())
}

#[test]
fn test_square_root() -> Result<()> {
    let a = Tensor::from_data(
        vec![4.0, 9.0, 16.0, 25.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.sqrt()?;
    assert_tensor_eq(&result, &[2.0, 3.0, 4.0, 5.0])?;
    Ok(())
}

#[test]
fn test_exponential() -> Result<()> {
    let a = Tensor::from_data(
        vec![0.0, 1.0],
        vec![2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.exp()?;
    let data = result.to_vec()?;
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - std::f32::consts::E).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_logarithm() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, std::f32::consts::E],
        vec![2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.log()?;
    let data = result.to_vec()?;
    assert!(data[0].abs() < 1e-6);
    assert!((data[1] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_relu_activation() -> Result<()> {
    let a = Tensor::from_data(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        vec![5],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.relu()?;
    assert_tensor_eq(&result, &[0.0, 0.0, 0.0, 1.0, 2.0])?;
    Ok(())
}

#[test]
fn test_sigmoid_activation() -> Result<()> {
    let a = Tensor::from_data(vec![0.0], vec![1], DataType::F32, TensorLayout::RowMajor)?;

    let result = a.sigmoid()?;
    let data = result.to_vec()?;
    assert!((data[0] - 0.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_tanh_activation() -> Result<()> {
    let a = Tensor::from_data(vec![0.0], vec![1], DataType::F32, TensorLayout::RowMajor)?;

    let result = a.tanh()?;
    let data = result.to_vec()?;
    assert!(data[0].abs() < 1e-6);
    Ok(())
}

#[test]
fn test_gelu_activation() -> Result<()> {
    let a = Tensor::from_data(
        vec![0.0, 1.0],
        vec![2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.gelu()?;
    let data = result.to_vec()?;
    assert!(data[0].abs() < 1e-1); // GELU(0) ≈ 0
    assert!(data[1] > 0.8); // GELU(1) ≈ 0.84
    Ok(())
}

#[test]
fn test_clamp_operation() -> Result<()> {
    let a = Tensor::from_data(
        vec![-5.0, -1.0, 0.0, 3.0, 10.0],
        vec![5],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.clamp(-2.0, 5.0)?;
    assert_tensor_eq(&result, &[-2.0, -1.0, 0.0, 3.0, 5.0])?;
    Ok(())
}

#[test]
fn test_clamp_invalid_range() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    // min > max should fail
    assert!(a.clamp(10.0, 5.0).is_err());
}

#[test]
fn test_division_by_zero_error() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )
    .unwrap();

    assert!(a.div_scalar(0.0).is_err());
}

#[test]
fn test_arithmetic_with_different_dtypes() -> Result<()> {
    // Test F32
    let a_f32 = Tensor::from_data(
        vec![1.0, 2.0],
        vec![2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let result_f32 = a_f32.add_scalar(5.0)?;
    assert_eq!(result_f32.dtype(), DataType::F32);

    // Test F16
    let a_f16 = Tensor::from_data(
        vec![1.0, 2.0],
        vec![2],
        DataType::F16,
        TensorLayout::RowMajor,
    )?;
    let result_f16 = a_f16.add_scalar(5.0)?;
    assert_eq!(result_f16.dtype(), DataType::F16);

    Ok(())
}

#[test]
fn test_zero_size_tensor_arithmetic() -> Result<()> {
    let a = Tensor::zeros(vec![0], DataType::F32, TensorLayout::RowMajor)?;
    let b = Tensor::zeros(vec![0], DataType::F32, TensorLayout::RowMajor)?;

    // Operations on zero-size tensors should work
    let result = a.add(&b)?;
    assert_eq!(result.numel(), 0);
    Ok(())
}

#[test]
fn test_large_tensor_operations() -> Result<()> {
    let a = Tensor::ones(vec![1000, 1000], DataType::F32, TensorLayout::RowMajor)?;
    let b = Tensor::ones(vec![1000, 1000], DataType::F32, TensorLayout::RowMajor)?;

    let result = a.add(&b)?;
    assert_eq!(result.shape(), vec![1000, 1000]);

    // Sample check a few values
    let data = result.to_vec()?;
    assert!((data[0] - 2.0).abs() < 1e-6);
    assert!((data[500] - 2.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_chained_operations() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // (a + 5) * 2 - 3
    let result = a.add_scalar(5.0)?.mul_scalar(2.0)?.sub_scalar(3.0)?;

    assert_tensor_eq(&result, &[9.0, 11.0, 13.0, 15.0])?;
    Ok(())
}

#[test]
fn test_broadcasting_multiple_dimensions() -> Result<()> {
    // Shape [2, 1, 3] + Shape [1, 4, 1] -> Shape [2, 4, 3]
    let a = Tensor::ones(vec![2, 1, 3], DataType::F32, TensorLayout::RowMajor)?;
    let b = Tensor::ones(vec![1, 4, 1], DataType::F32, TensorLayout::RowMajor)?;

    let result = a.add(&b)?;
    assert_eq!(result.shape(), vec![2, 4, 3]);

    let data = result.to_vec()?;
    assert!(data.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    Ok(())
}
