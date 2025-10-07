//! Continual Learning Engine - Adaptive Learning Without Catastrophic Forgetting
//!
//! Implements mechanisms for continual learning inspired by neuroscience:
//! - **Multi-timescale Learning**: Fast and slow weight adaptation
//! - **Elastic Weight Consolidation (EWC)**: Protecting important weights
//! - **Experience Replay**: Rehearsing important experiences
//!
//! ## Key Concepts
//!
//! - **Fast Weights**: High learning rate, quickly adapt to new tasks
//! - **Slow Weights**: Low learning rate, stable long-term knowledge
//! - **Importance Weighting**: Protect weights critical for previous tasks
//!
//! ```text
//! New Experience
//!       ↓
//! Fast Weights (Quick Adaptation)
//!       ↓
//! [EWC Protection Check]
//!       ↓
//! Slow Weights (Gradual Integration)
//!       ↓
//! Experience Replay (Rehearsal)
//! ```

pub mod ewc;
pub mod replay;
pub mod timescales;

pub use ewc::{ElasticWeightConsolidation, ImportanceWeights};
pub use replay::{Experience, ExperienceReplay, ReplayStrategy};
pub use timescales::{MultiTimescaleLearner, TimescaleConfig, WeightUpdate};

use ronn_core::tensor::Tensor;
use thiserror::Error;

/// Errors that can occur in the learning system
#[derive(Error, Debug)]
pub enum LearningError {
    #[error("Multi-timescale learning error: {0}")]
    Timescale(String),

    #[error("EWC error: {0}")]
    EWC(String),

    #[error("Replay error: {0}")]
    Replay(String),

    #[error("Core error: {0}")]
    Core(#[from] ronn_core::error::CoreError),
}

pub type Result<T> = std::result::Result<T, LearningError>;

/// Main continual learning engine
pub struct ContinualLearningEngine {
    timescale_learner: MultiTimescaleLearner,
    ewc: ElasticWeightConsolidation,
    replay_buffer: ExperienceReplay,
    config: LearningConfig,
    stats: LearningStats,
}

impl ContinualLearningEngine {
    /// Create a new continual learning engine
    pub fn new(config: LearningConfig) -> Self {
        Self {
            timescale_learner: MultiTimescaleLearner::new(config.timescale.clone()),
            ewc: ElasticWeightConsolidation::new(config.ewc_lambda),
            replay_buffer: ExperienceReplay::new(config.replay_capacity, config.replay_strategy),
            config,
            stats: LearningStats::default(),
        }
    }

    /// Learn from a new experience
    pub fn learn(
        &mut self,
        input: Tensor,
        target: Tensor,
        importance: f64,
    ) -> Result<LearningResult> {
        self.stats.total_updates += 1;

        // 1. Store experience for replay
        let experience = Experience {
            input: input.clone(),
            target: target.clone(),
            importance,
            timestamp: current_timestamp(),
        };
        self.replay_buffer.store(experience)?;

        // 2. Perform multi-timescale update
        let weight_update = self.timescale_learner.compute_update(&input, &target)?;

        // 3. Apply EWC constraints (protect important weights)
        let protected_update = self.ewc.constrain_update(&weight_update)?;

        // 4. Apply the update
        let applied = self.timescale_learner.apply_update(&protected_update)?;

        // 5. Periodically replay experiences
        if self.stats.total_updates % self.config.replay_frequency == 0 {
            self.replay_experiences()?;
        }

        Ok(LearningResult {
            fast_weight_change: applied.fast_magnitude,
            slow_weight_change: applied.slow_magnitude,
            ewc_penalty: protected_update.ewc_penalty,
        })
    }

    /// Consolidate task (mark current weights as important)
    pub fn consolidate_task(&mut self, task_data: &[(Tensor, Tensor)]) -> Result<()> {
        self.stats.tasks_learned += 1;

        // Compute importance of current weights for this task
        self.ewc.compute_importance(task_data)?;

        // Consolidate fast weights into slow weights
        self.timescale_learner.consolidate()?;

        Ok(())
    }

    /// Replay experiences from buffer
    fn replay_experiences(&mut self) -> Result<()> {
        let experiences = self.replay_buffer.sample(self.config.replay_batch_size)?;

        for exp in experiences {
            // Re-learn from experience (with reduced learning rate)
            let update = self
                .timescale_learner
                .compute_update(&exp.input, &exp.target)?;
            let scaled_update = update.scale(0.5); // Reduce impact of replay
            self.timescale_learner.apply_update(&scaled_update)?;
        }

        self.stats.replay_count += 1;

        Ok(())
    }

    /// Get learning statistics
    pub fn stats(&self) -> &LearningStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = LearningStats::default();
    }
}

/// Configuration for continual learning
#[derive(Debug, Clone)]
pub struct LearningConfig {
    pub timescale: TimescaleConfig,
    pub ewc_lambda: f64,
    pub replay_capacity: usize,
    pub replay_strategy: ReplayStrategy,
    pub replay_frequency: u64,
    pub replay_batch_size: usize,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            timescale: TimescaleConfig::default(),
            ewc_lambda: 0.4,
            replay_capacity: 1000,
            replay_strategy: ReplayStrategy::Importance,
            replay_frequency: 10,
            replay_batch_size: 32,
        }
    }
}

/// Result of a learning update
#[derive(Debug, Clone)]
pub struct LearningResult {
    pub fast_weight_change: f64,
    pub slow_weight_change: f64,
    pub ewc_penalty: f64,
}

/// Statistics about learning progress
#[derive(Debug, Clone, Default)]
pub struct LearningStats {
    pub total_updates: u64,
    pub tasks_learned: u64,
    pub replay_count: u64,
}

/// Get current timestamp in milliseconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::types::{DataType, TensorLayout};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    #[test]
    fn test_engine_creation() {
        let engine = ContinualLearningEngine::new(LearningConfig::default());
        assert_eq!(engine.stats().total_updates, 0);
    }

    #[test]
    fn test_learning_update() -> Result<()> {
        let mut engine = ContinualLearningEngine::new(LearningConfig::default());

        let input = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![1, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let target = Tensor::from_data(
            vec![0.5f32, 0.5],
            vec![1, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let result = engine.learn(input, target, 0.8)?;

        assert_eq!(engine.stats().total_updates, 1);
        assert!(result.fast_weight_change >= 0.0);

        Ok(())
    }

    #[test]
    fn test_task_consolidation() -> Result<()> {
        let mut engine = ContinualLearningEngine::new(LearningConfig::default());

        // Simulate learning a task
        let task_data = vec![(
            Tensor::from_data(
                vec![1.0f32, 2.0],
                vec![1, 2],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
            Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?,
        )];

        engine.consolidate_task(&task_data)?;

        assert_eq!(engine.stats().tasks_learned, 1);

        Ok(())
    }

    #[test]
    fn test_multiple_updates_trigger_replay() -> Result<()> {
        let config = LearningConfig {
            replay_frequency: 5,
            ..Default::default()
        };

        let mut engine = ContinualLearningEngine::new(config);

        // Perform multiple updates
        for i in 0..10 {
            let input = Tensor::from_data(
                vec![i as f32, (i + 1) as f32],
                vec![1, 2],
                DataType::F32,
                TensorLayout::RowMajor,
            )?;
            let target = Tensor::from_data(
                vec![0.5f32],
                vec![1, 1],
                DataType::F32,
                TensorLayout::RowMajor,
            )?;

            engine.learn(input, target, 0.5)?;
        }

        // Should have triggered replay
        assert!(engine.stats().replay_count > 0);

        Ok(())
    }
}
