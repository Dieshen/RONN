use super::{OnnxOperator, Result};
use crate::error::OnnxError;
use ronn_core::tensor::Tensor;
use ronn_core::NodeAttribute;
use std::collections::HashMap;

// Conv2d: 2D convolution
pub struct Conv2dOp;

impl OnnxOperator for Conv2dOp {
    fn op_type(&self) -> &str {
        "Conv"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Conv expects at least 2 inputs (input, weight), got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let weight = inputs[1];
        let bias = if inputs.len() > 2 {
            Some(inputs[2])
        } else {
            None
        };

        // Extract attributes
        let strides = Self::get_ints_attr(attributes, "strides", vec![1, 1]);
        let pads = Self::get_ints_attr(attributes, "pads", vec![0, 0, 0, 0]);
        let dilations = Self::get_ints_attr(attributes, "dilations", vec![1, 1]);
        let group = Self::get_int_attr(attributes, "group", 1);

        // Convert to usize
        let strides_usize: Vec<usize> = strides.iter().map(|&x| x as usize).collect();
        let pads_usize: Vec<usize> = pads.iter().map(|&x| x as usize).collect();
        let dilations_usize: Vec<usize> = dilations.iter().map(|&x| x as usize).collect();

        // Perform convolution
        let result = input.conv2d(
            weight,
            bias,
            &strides_usize,
            &pads_usize,
            &dilations_usize,
            group as usize,
        )?;
        Ok(vec![result])
    }
}

impl Conv2dOp {
    fn get_ints_attr(
        attrs: &HashMap<String, NodeAttribute>,
        name: &str,
        default: Vec<i64>,
    ) -> Vec<i64> {
        if let Some(NodeAttribute::IntArray(v)) = attrs.get(name) {
            v.clone()
        } else {
            default
        }
    }

    fn get_int_attr(attrs: &HashMap<String, NodeAttribute>, name: &str, default: i64) -> i64 {
        if let Some(NodeAttribute::Int(v)) = attrs.get(name) {
            *v
        } else {
            default
        }
    }
}

// MaxPool: max pooling operation
pub struct MaxPoolOp;

impl OnnxOperator for MaxPoolOp {
    fn op_type(&self) -> &str {
        "MaxPool"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "MaxPool expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract attributes
        let kernel_shape = Self::get_ints_attr(attributes, "kernel_shape", vec![2, 2]);
        let strides = Self::get_ints_attr(attributes, "strides", vec![1, 1]);
        let pads = Self::get_ints_attr(attributes, "pads", vec![0, 0, 0, 0]);

        // Convert to usize
        let kernel_shape_usize: Vec<usize> = kernel_shape.iter().map(|&x| x as usize).collect();
        let strides_usize: Vec<usize> = strides.iter().map(|&x| x as usize).collect();
        let pads_usize: Vec<usize> = pads.iter().map(|&x| x as usize).collect();

        let result = inputs[0].max_pool2d(&kernel_shape_usize, &strides_usize, &pads_usize)?;
        Ok(vec![result])
    }
}

impl MaxPoolOp {
    fn get_ints_attr(
        attrs: &HashMap<String, NodeAttribute>,
        name: &str,
        default: Vec<i64>,
    ) -> Vec<i64> {
        if let Some(NodeAttribute::IntArray(v)) = attrs.get(name) {
            v.clone()
        } else {
            default
        }
    }
}

// AveragePool: average pooling operation
pub struct AvgPoolOp;

impl OnnxOperator for AvgPoolOp {
    fn op_type(&self) -> &str {
        "AveragePool"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidGraph(format!(
                "AveragePool expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract attributes
        let kernel_shape = Self::get_ints_attr(attributes, "kernel_shape", vec![2, 2]);
        let strides = Self::get_ints_attr(attributes, "strides", vec![1, 1]);
        let pads = Self::get_ints_attr(attributes, "pads", vec![0, 0, 0, 0]);

        // Convert to usize
        let kernel_shape_usize: Vec<usize> = kernel_shape.iter().map(|&x| x as usize).collect();
        let strides_usize: Vec<usize> = strides.iter().map(|&x| x as usize).collect();
        let pads_usize: Vec<usize> = pads.iter().map(|&x| x as usize).collect();

        let result = inputs[0].avg_pool2d(&kernel_shape_usize, &strides_usize, &pads_usize)?;
        Ok(vec![result])
    }
}

impl AvgPoolOp {
    fn get_ints_attr(
        attrs: &HashMap<String, NodeAttribute>,
        name: &str,
        default: Vec<i64>,
    ) -> Vec<i64> {
        if let Some(NodeAttribute::IntArray(v)) = attrs.get(name) {
            v.clone()
        } else {
            default
        }
    }
}

// BatchNormalization: batch normalization
pub struct BatchNormOp;

impl OnnxOperator for BatchNormOp {
    fn op_type(&self) -> &str {
        "BatchNormalization"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() < 5 {
            return Err(OnnxError::InvalidGraph(format!(
                "BatchNormalization expects 5 inputs (input, scale, bias, mean, var), got {}",
                inputs.len()
            )));
        }

        let input = inputs[0];
        let scale = inputs[1];
        let bias = inputs[2];
        let mean = inputs[3];
        let var = inputs[4];

        // Extract epsilon attribute
        let epsilon = if let Some(NodeAttribute::Float(e)) = attributes.get("epsilon") {
            *e
        } else {
            1e-5
        };

        let result = input.batch_norm(scale, bias, mean, var, epsilon as f32)?;
        Ok(vec![result])
    }
}
