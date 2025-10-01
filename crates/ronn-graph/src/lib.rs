// Graph optimization pipeline for RONN runtime
//
// This crate provides graph-level optimizations including:
// - Constant folding
// - Dead code elimination
// - Node fusion
// - Layout optimization
// - Provider-specific optimization passes

mod error;
mod optimizer;
mod passes;

pub use error::{OptimizationError, Result};
pub use optimizer::{OptimizationLevel, Optimizer, PassManager};
pub use passes::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = Optimizer::new(OptimizationLevel::O2);
        assert!(optimizer.pass_count() > 0);
    }
}
