//! Comprehensive tests for tensor matrix operations.
//!
//! This module tests all matrix operations including:
//! - Matrix multiplication
//! - Transpose
//! - Linear algebra operations (determinant, inverse, trace)
//! - Batch operations

mod test_utils;

use anyhow::Result;
use ronn_core::{DataType, MatrixOps, Tensor, TensorLayout};
use test_utils::*;

#[test]
fn test_matrix_multiplication_2x2() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![2.0, 0.0, 1.0, 2.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.matmul(&b)?;
    // [[1*2+2*1, 1*0+2*2], [3*2+4*1, 3*0+4*2]] = [[4, 4], [10, 8]]
    assert_tensor_eq(&result, &[4.0, 4.0, 10.0, 8.0])?;
    Ok(())
}

#[test]
fn test_matrix_multiplication_rectangular() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;
    let b = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.matmul(&b)?;
    assert_eq!(result.shape(), vec![2, 2]);
    // [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
    // = [[22, 28], [49, 64]]
    assert_tensor_eq(&result, &[22.0, 28.0, 49.0, 64.0])?;
    Ok(())
}

#[test]
fn test_matrix_multiplication_incompatible_dims() {
    let a = Tensor::from_data(
        vec![1.0, 2.0],
        vec![2, 1],
        DataType::F32,
        TensorLayout::RowMajor,
    ).unwrap();
    let b = Tensor::from_data(
        vec![1.0, 2.0, 3.0],
        vec![3, 1],
        DataType::F32,
        TensorLayout::RowMajor,
    ).unwrap();

    // 2x1 @ 3x1 should fail (inner dimensions don't match)
    assert!(a.matmul(&b).is_err());
}

#[test]
fn test_transpose_2d() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = MatrixOps::transpose(&a)?;
    assert_eq!(result.shape(), vec![3, 2]);
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
    assert_tensor_eq(&result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    Ok(())
}

#[test]
fn test_transpose_dims() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.transpose_dims(0, 2)?;
    assert_eq!(result.shape(), vec![2, 2, 2]);
    Ok(())
}

#[test]
fn test_transpose_invalid_dims() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    ).unwrap();

    // Out of bounds dimensions
    assert!(a.transpose_dims(5, 6).is_err());
}

#[test]
fn test_identity_matrix() -> Result<()> {
    let identity = Tensor::eye(3, DataType::F32, TensorLayout::RowMajor)?;

    assert_eq!(identity.shape(), vec![3, 3]);
    assert_tensor_eq(&identity, &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ])?;
    Ok(())
}

#[test]
fn test_identity_matrix_sizes() -> Result<()> {
    for size in [1, 2, 3, 5, 10] {
        let identity = Tensor::eye(size, DataType::F32, TensorLayout::RowMajor)?;
        assert_eq!(identity.shape(), vec![size, size]);

        let data = identity.to_vec()?;
        for i in 0..size {
            for j in 0..size {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(data[i * size + j], expected);
            }
        }
    }
    Ok(())
}

#[test]
fn test_diagonal_extraction() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let diag = a.diagonal()?;
    assert_tensor_eq(&diag, &[1.0, 4.0])?;
    Ok(())
}

#[test]
fn test_diagonal_rectangular() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let diag = a.diagonal()?;
    assert_tensor_eq(&diag, &[1.0, 5.0])?; // min(2, 3) = 2 elements
    Ok(())
}

#[test]
fn test_trace_2x2() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let trace = a.trace()?;
    assert_tensor_eq(&trace, &[5.0])?; // 1 + 4 = 5
    Ok(())
}

#[test]
fn test_trace_3x3() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        vec![3, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let trace = a.trace()?;
    assert_tensor_eq(&trace, &[6.0])?; // 1 + 2 + 3 = 6
    Ok(())
}

#[test]
fn test_determinant_1x1() -> Result<()> {
    let a = Tensor::from_data(
        vec![5.0],
        vec![1, 1],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let det = a.det()?;
    assert_tensor_eq(&det, &[5.0])?;
    Ok(())
}

#[test]
fn test_determinant_2x2() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let det = a.det()?;
    // det = 1*4 - 2*3 = -2
    let data = det.to_vec()?;
    assert!((data[0] + 2.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_inverse_2x2() -> Result<()> {
    let a = Tensor::from_data(
        vec![4.0, 7.0, 2.0, 6.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let inv = a.inverse()?;

    // Verify A * A^-1 = I
    let identity = a.matmul(&inv)?;
    let identity_data = identity.to_vec()?;

    assert!((identity_data[0] - 1.0).abs() < 1e-5);
    assert!(identity_data[1].abs() < 1e-5);
    assert!(identity_data[2].abs() < 1e-5);
    assert!((identity_data[3] - 1.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_inverse_singular_matrix() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 2.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    ).unwrap();

    // Singular matrix (det = 0) should fail
    assert!(a.inverse().is_err());
}

#[test]
fn test_inverse_non_square() {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    ).unwrap();

    assert!(a.inverse().is_err());
}

#[test]
fn test_frobenius_norm() -> Result<()> {
    let a = Tensor::from_data(
        vec![3.0, 4.0],
        vec![2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let norm = a.frobenius_norm()?;
    assert_tensor_eq(&norm, &[5.0])?; // sqrt(9 + 16) = 5
    Ok(())
}

#[test]
fn test_frobenius_norm_matrix() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 0.0, 0.0, 1.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let norm = a.frobenius_norm()?;
    let data = norm.to_vec()?;
    assert!((data[0] - f32::sqrt(2.0)).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_diag_embed() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0],
        vec![3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let diag_matrix = a.diag_embed()?;
    assert_eq!(diag_matrix.shape(), vec![3, 3]);
    assert_tensor_eq(&diag_matrix, &[
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ])?;
    Ok(())
}

#[test]
fn test_batch_matmul_3d() -> Result<()> {
    // Batch of 2 matrices, each 2x3
    let a = Tensor::from_data(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  // First matrix
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Second matrix
        ],
        vec![2, 2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Batch of 2 matrices, each 3x2
    let b = Tensor::from_data(
        vec![
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0,  // First matrix (identity-like)
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0,  // Second matrix
        ],
        vec![2, 3, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = a.batch_matmul(&b)?;
    assert_eq!(result.shape(), vec![2, 2, 2]);
    Ok(())
}

#[test]
fn test_matmul_identity_property() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let identity = Tensor::eye(2, DataType::F32, TensorLayout::RowMajor)?;

    let result = a.matmul(&identity)?;
    assert_tensor_approx_eq(&result, &a, 1e-6)?;
    Ok(())
}

#[test]
fn test_transpose_transpose_identity() -> Result<()> {
    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    let result = MatrixOps::transpose(&MatrixOps::transpose(&a)?)?;
    assert_tensor_approx_eq(&result, &a, 1e-6)?;
    Ok(())
}

#[test]
fn test_matmul_transpose_property() -> Result<()> {
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

    // (AB)^T = B^T A^T
    let ab = a.matmul(&b)?;
    let ab_t = MatrixOps::transpose(&ab)?;

    let b_t = MatrixOps::transpose(&b)?;
    let a_t = MatrixOps::transpose(&a)?;
    let bt_at = b_t.matmul(&a_t)?;

    assert_tensor_approx_eq(&ab_t, &bt_at, 1e-6)?;
    Ok(())
}

#[test]
fn test_large_matrix_multiplication() -> Result<()> {
    let a = Tensor::ones(vec![100, 200], DataType::F32, TensorLayout::RowMajor)?;
    let b = Tensor::ones(vec![200, 150], DataType::F32, TensorLayout::RowMajor)?;

    let result = a.matmul(&b)?;
    assert_eq!(result.shape(), vec![100, 150]);

    // Each element should be 200.0 (sum of 200 ones)
    let data = result.to_vec()?;
    assert!((data[0] - 200.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_square_matrix_operations() -> Result<()> {
    let a = Tensor::from_data(
        vec![2.0, 1.0, 1.0, 2.0],
        vec![2, 2],
        DataType::F32,
        TensorLayout::RowMajor,
    )?;

    // Test trace
    let trace = a.trace()?;
    assert_tensor_eq(&trace, &[4.0])?;

    // Test determinant
    let det = a.det()?;
    let det_data = det.to_vec()?;
    assert!((det_data[0] - 3.0).abs() < 1e-6);

    // Test diagonal
    let diag = a.diagonal()?;
    assert_tensor_eq(&diag, &[2.0, 2.0])?;

    Ok(())
}
