//! Tensor operations module.
//!
//! This module provides all tensor operations including arithmetic, matrix,
//! shape manipulation, and reduction operations using Candle backend.

pub mod arithmetic;
pub mod matrix;
pub mod reduction;
pub mod shape;

// Re-export all operations
pub use arithmetic::*;
pub use matrix::*;
pub use reduction::*;
pub use shape::*;
