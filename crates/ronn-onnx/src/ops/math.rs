use super::{OnnxOperator, Result};
use crate::error::OnnxError;
use ronn_core::ops::{ArithmeticOps, MatrixOps};
use ronn_core::tensor::Tensor;
use ronn_core::NodeAttribute;
use std::collections::HashMap;

// Add: element-wise addition with broadcasting
pub struct AddOp;

impl OnnxOperator for AddOp {
    fn op_type(&self) -> &str {
        "Add"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Add expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].add(inputs[1])?;
        Ok(vec![result])
    }
}

// Sub: element-wise subtraction with broadcasting
pub struct SubOp;

impl OnnxOperator for SubOp {
    fn op_type(&self) -> &str {
        "Sub"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Sub expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].sub(inputs[1])?;
        Ok(vec![result])
    }
}

// Mul: element-wise multiplication with broadcasting
pub struct MulOp;

impl OnnxOperator for MulOp {
    fn op_type(&self) -> &str {
        "Mul"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Mul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].mul(inputs[1])?;
        Ok(vec![result])
    }
}

// Div: element-wise division with broadcasting
pub struct DivOp;

impl OnnxOperator for DivOp {
    fn op_type(&self) -> &str {
        "Div"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "Div expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].div(inputs[1])?;
        Ok(vec![result])
    }
}

// MatMul: matrix multiplication
pub struct MatMulOp;

impl OnnxOperator for MatMulOp {
    fn op_type(&self) -> &str {
        "MatMul"
    }

    fn execute(
        &self,
        inputs: &[&Tensor],
        _attributes: &HashMap<String, NodeAttribute>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidGraph(format!(
                "MatMul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let result = inputs[0].matmul(inputs[1])?;
        Ok(vec![result])
    }
}
