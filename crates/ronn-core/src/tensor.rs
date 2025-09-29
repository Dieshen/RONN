//! Tensor implementation with Candle backend integration.
//!
//! This module provides the core Tensor type for RONN with seamless integration
//! to the Candle tensor library for high-performance operations and GPU acceleration.

use crate::types::{DataType, Tensor as RonnTensor, TensorLayout};
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Shape, Tensor as CandleTensor};

/// Enhanced Tensor implementation with Candle backend.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The underlying Candle tensor for computation.
    candle_tensor: CandleTensor,
    /// Original data type specification.
    dtype: DataType,
    /// Memory layout preference.
    layout: TensorLayout,
}

impl Tensor {
    /// Create a new tensor from raw data.
    ///
    /// # Arguments
    /// * `data` - Raw tensor data
    /// * `shape` - Tensor dimensions
    /// * `dtype` - Data type specification
    /// * `layout` - Memory layout preference
    ///
    /// # Example
    /// ```rust
    /// use ronn_core::tensor::Tensor;
    /// use ronn_core::types::{DataType, TensorLayout};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_data(
        data: Vec<f32>,
        shape: Vec<usize>,
        dtype: DataType,
        layout: TensorLayout,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let candle_shape = Shape::from_dims(&shape);

        let candle_tensor = match dtype {
            DataType::F32 => CandleTensor::from_vec(data, candle_shape, &device)?,
            DataType::F16 => {
                let f16_data: Vec<half::f16> = data.into_iter().map(half::f16::from_f32).collect();
                CandleTensor::from_vec(f16_data, candle_shape, &device)?
            }
            DataType::F64 => {
                let f64_data: Vec<f64> = data.into_iter().map(|x| x as f64).collect();
                CandleTensor::from_vec(f64_data, candle_shape, &device)?
            }
            DataType::U8 => {
                let u8_data: Vec<u8> = data.into_iter().map(|x| x as u8).collect();
                CandleTensor::from_vec(u8_data, candle_shape, &device)?
            }
            DataType::U32 => {
                let u32_data: Vec<u32> = data.into_iter().map(|x| x as u32).collect();
                CandleTensor::from_vec(u32_data, candle_shape, &device)?
            }
            // For unsupported types, convert to F32
            DataType::I8 | DataType::I32 | DataType::Bool => {
                CandleTensor::from_vec(data, candle_shape, &device)?
            }
        };

        Ok(Self {
            candle_tensor,
            dtype,
            layout,
        })
    }

    /// Create a tensor filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - Tensor dimensions
    /// * `dtype` - Data type specification
    /// * `layout` - Memory layout preference
    pub fn zeros(shape: Vec<usize>, dtype: DataType, layout: TensorLayout) -> Result<Self> {
        let device = Device::Cpu;
        let candle_dtype = dtype_to_candle(&dtype)?;
        let candle_shape = Shape::from_dims(&shape);

        let candle_tensor = CandleTensor::zeros(candle_shape, candle_dtype, &device)?;

        Ok(Self {
            candle_tensor,
            dtype,
            layout,
        })
    }

    /// Create a tensor filled with ones.
    ///
    /// # Arguments
    /// * `shape` - Tensor dimensions
    /// * `dtype` - Data type specification
    /// * `layout` - Memory layout preference
    pub fn ones(shape: Vec<usize>, dtype: DataType, layout: TensorLayout) -> Result<Self> {
        let device = Device::Cpu;
        let candle_dtype = dtype_to_candle(&dtype)?;
        let candle_shape = Shape::from_dims(&shape);

        let candle_tensor = CandleTensor::ones(candle_shape, candle_dtype, &device)?;

        Ok(Self {
            candle_tensor,
            dtype,
            layout,
        })
    }

    /// Create a tensor with random values from a uniform distribution.
    pub fn rand(shape: Vec<usize>, dtype: DataType, layout: TensorLayout) -> Result<Self> {
        let device = Device::Cpu;
        let _candle_dtype = dtype_to_candle(&dtype)?;
        let candle_shape = Shape::from_dims(&shape);

        let candle_tensor = CandleTensor::rand(0.0, 1.0, candle_shape, &device)?;

        Ok(Self {
            candle_tensor,
            dtype,
            layout,
        })
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.candle_tensor.dims().to_vec()
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get the memory layout of the tensor.
    pub fn layout(&self) -> TensorLayout {
        self.layout
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.candle_tensor.dims().len()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.candle_tensor.elem_count()
    }

    /// Get the device where the tensor is stored.
    pub fn device(&self) -> &Device {
        self.candle_tensor.device()
    }

    /// Convert tensor to CPU device.
    pub fn to_cpu(&self) -> Result<Self> {
        let cpu_tensor = self.candle_tensor.to_device(&Device::Cpu)?;
        Ok(Self {
            candle_tensor: cpu_tensor,
            dtype: self.dtype,
            layout: self.layout,
        })
    }

    /// Convert tensor to GPU device (if available).
    #[cfg(feature = "gpu")]
    pub fn to_gpu(&self, device_id: usize) -> Result<Self> {
        let gpu_device = Device::new_cuda(device_id)?;
        let gpu_tensor = self.candle_tensor.to_device(&gpu_device)?;
        Ok(Self {
            candle_tensor: gpu_tensor,
            dtype: self.dtype,
            layout: self.layout,
        })
    }

    /// Extract data as a vector of f32 values.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        // Flatten the tensor first if it's multi-dimensional
        let flattened = if self.candle_tensor.dims().len() > 1 {
            self.candle_tensor.flatten_all()?
        } else {
            self.candle_tensor.clone()
        };

        match self.dtype {
            DataType::F32 | DataType::I8 | DataType::I32 | DataType::Bool => {
                let data: Vec<f32> = flattened.to_vec1()?;
                Ok(data)
            }
            DataType::F16 => {
                let data: Vec<half::f16> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x.to_f32()).collect())
            }
            DataType::F64 => {
                let data: Vec<f64> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
            DataType::U8 => {
                let data: Vec<u8> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
            DataType::U32 => {
                let data: Vec<u32> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
        }
    }

    /// Get the underlying Candle tensor for advanced operations.
    pub fn candle_tensor(&self) -> &CandleTensor {
        &self.candle_tensor
    }

    /// Create a Tensor from a Candle tensor.
    pub fn from_candle(candle_tensor: CandleTensor, dtype: DataType, layout: TensorLayout) -> Self {
        Self {
            candle_tensor,
            dtype,
            layout,
        }
    }

    /// Check if tensor shapes are broadcastable.
    pub fn is_broadcastable_with(&self, other: &Tensor) -> bool {
        let shape1 = self.shape();
        let shape2 = other.shape();

        // Pad shorter shape with 1s on the left
        let max_len = shape1.len().max(shape2.len());
        let mut padded1 = vec![1; max_len - shape1.len()];
        let mut padded2 = vec![1; max_len - shape2.len()];
        padded1.extend(shape1);
        padded2.extend(shape2);

        // Check compatibility dimension by dimension
        for (d1, d2) in padded1.iter().zip(padded2.iter()) {
            if *d1 != *d2 && *d1 != 1 && *d2 != 1 {
                return false;
            }
        }
        true
    }

    /// Compute broadcast shape for two tensors.
    pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
        let max_len = shape1.len().max(shape2.len());
        let mut padded1 = vec![1; max_len - shape1.len()];
        let mut padded2 = vec![1; max_len - shape2.len()];
        padded1.extend(shape1);
        padded2.extend(shape2);

        let mut result = Vec::with_capacity(max_len);
        for (d1, d2) in padded1.iter().zip(padded2.iter()) {
            match (d1, d2) {
                (1, d) | (d, 1) => result.push(*d),
                (d1, d2) if d1 == d2 => result.push(*d1),
                (d1, d2) => {
                    return Err(anyhow!(
                        "Cannot broadcast shapes: dimension {} vs {}",
                        d1,
                        d2
                    ))
                }
            }
        }
        Ok(result)
    }
}

/// Convert RONN DataType to Candle DType.
fn dtype_to_candle(dtype: &DataType) -> Result<DType> {
    match dtype {
        DataType::F32 => Ok(DType::F32),
        DataType::F16 => Ok(DType::F16),
        DataType::F64 => Ok(DType::F64),
        DataType::U8 => Ok(DType::U8),
        DataType::U32 => Ok(DType::U32),
        // For unsupported types, use F32
        DataType::I8 | DataType::I32 | DataType::Bool => Ok(DType::F32),
    }
}

/// Convert Candle DType to RONN DataType.
#[allow(dead_code)]
fn dtype_from_candle(dtype: DType) -> DataType {
    match dtype {
        DType::F32 => DataType::F32,
        DType::F16 => DataType::F16,
        DType::U8 => DataType::U8,
        DType::U32 => DataType::U32,
        DType::F64 => DataType::F64,
        _ => DataType::F32, // Default fallback
    }
}

/// Convert legacy RonnTensor to new Tensor implementation.
impl From<RonnTensor> for Tensor {
    fn from(legacy: RonnTensor) -> Self {
        Self::from_data(legacy.data, legacy.shape, legacy.dtype, legacy.layout)
            .expect("Failed to convert legacy tensor")
    }
}

/// Convert new Tensor to legacy RonnTensor for compatibility.
impl From<Tensor> for RonnTensor {
    fn from(tensor: Tensor) -> Self {
        let data = tensor.to_vec().expect("Failed to extract tensor data");
        Self {
            data,
            shape: tensor.shape(),
            dtype: tensor.dtype,
            layout: tensor.layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(
            data.clone(),
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.dtype(), DataType::F32);
        assert_eq!(tensor.numel(), 4);

        let extracted = tensor.to_vec()?;
        assert_eq!(extracted, data);

        Ok(())
    }

    #[test]
    fn test_zeros_and_ones() -> Result<()> {
        let zeros = Tensor::zeros(vec![3, 3], DataType::F32, TensorLayout::RowMajor)?;
        let zeros_data = zeros.to_vec()?;
        assert!(zeros_data.iter().all(|&x| x == 0.0));

        let ones = Tensor::ones(vec![2, 3], DataType::F32, TensorLayout::RowMajor)?;
        let ones_data = ones.to_vec()?;
        assert!(ones_data.iter().all(|&x| x == 1.0));

        Ok(())
    }

    #[test]
    fn test_broadcasting() {
        // Compatible shapes
        assert_eq!(
            Tensor::broadcast_shape(&[3, 1], &[1, 4]).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            Tensor::broadcast_shape(&[2, 3, 1], &[1, 4]).unwrap(),
            vec![2, 3, 4]
        );

        // Incompatible shapes
        assert!(Tensor::broadcast_shape(&[3, 2], &[2, 3]).is_err());
    }

    #[test]
    fn test_broadcastable_check() -> Result<()> {
        let tensor1 = Tensor::zeros(vec![3, 1], DataType::F32, TensorLayout::RowMajor)?;
        let tensor2 = Tensor::zeros(vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        let tensor3 = Tensor::zeros(vec![2, 3], DataType::F32, TensorLayout::RowMajor)?;

        assert!(tensor1.is_broadcastable_with(&tensor2));
        assert!(!tensor1.is_broadcastable_with(&tensor3));

        Ok(())
    }

    #[test]
    fn test_data_type_conversions() -> Result<()> {
        // Test F16 conversion
        let data = vec![1.5, 2.5, 3.5, 4.5];
        let tensor_f16 = Tensor::from_data(
            data.clone(),
            vec![2, 2],
            DataType::F16,
            TensorLayout::RowMajor,
        )?;
        let extracted_f16 = tensor_f16.to_vec()?;

        // F16 has limited precision, so we check with tolerance
        for (original, extracted) in data.iter().zip(extracted_f16.iter()) {
            assert!((original - extracted).abs() < 0.01);
        }

        // Test I8 conversion
        let int_data = vec![1.0, -2.0, 3.0, -4.0];
        let tensor_i8 =
            Tensor::from_data(int_data, vec![2, 2], DataType::I8, TensorLayout::RowMajor)?;
        let extracted_i8 = tensor_i8.to_vec()?;
        assert_eq!(extracted_i8, vec![1.0, -2.0, 3.0, -4.0]);

        Ok(())
    }

    #[test]
    fn test_device_operations() -> Result<()> {
        let tensor = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor)?;

        // Should be on CPU by default
        assert!(matches!(tensor.device(), Device::Cpu));

        // CPU conversion should work
        let cpu_tensor = tensor.to_cpu()?;
        assert!(matches!(cpu_tensor.device(), Device::Cpu));

        Ok(())
    }
}
