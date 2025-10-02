/// Common utilities and helpers for integration tests with real ONNX models.
///
/// This module provides:
/// - Reference output comparison utilities
/// - Accuracy validation functions
/// - Performance measurement helpers
/// - Test fixture loading

use std::path::Path;

/// Tolerance for floating-point comparisons in accuracy tests
pub const FP32_TOLERANCE: f32 = 1e-5;
pub const FP16_TOLERANCE: f32 = 1e-3;

/// Check if a model file exists and is accessible
pub fn model_exists(model_name: &str) -> bool {
    let model_path = format!("../../models/{}", model_name);
    Path::new(&model_path).exists()
}

/// Get the absolute path to a model file
pub fn model_path(model_name: &str) -> String {
    format!("../../models/{}", model_name)
}

/// Get the path to expected reference outputs
pub fn reference_output_path(test_name: &str) -> String {
    format!("../fixtures/expected_outputs/{}.json", test_name)
}

/// Compare two floating-point values with tolerance
pub fn approx_equal(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

/// Compare two float arrays element-wise with tolerance
pub fn compare_f32_arrays(actual: &[f32], expected: &[f32], tolerance: f32) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "Array length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        ));
    }

    let mut max_error = 0.0_f32;
    let mut error_count = 0;

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > tolerance {
            error_count += 1;
            max_error = max_error.max(diff);

            // Report first few errors for debugging
            if error_count <= 5 {
                eprintln!(
                    "  Error at index {}: actual={}, expected={}, diff={}",
                    i, a, e, diff
                );
            }
        }
    }

    if error_count > 0 {
        Err(format!(
            "Accuracy validation failed: {}/{} elements exceeded tolerance (max error: {})",
            error_count,
            actual.len(),
            max_error
        ))
    } else {
        Ok(())
    }
}

/// Calculate mean absolute error between two arrays
pub fn mean_absolute_error(actual: &[f32], expected: &[f32]) -> f32 {
    if actual.len() != expected.len() {
        return f32::INFINITY;
    }

    let sum: f32 = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .sum();

    sum / actual.len() as f32
}

/// Calculate mean squared error between two arrays
pub fn mean_squared_error(actual: &[f32], expected: &[f32]) -> f32 {
    if actual.len() != expected.len() {
        return f32::INFINITY;
    }

    let sum: f32 = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| {
            let diff = a - e;
            diff * diff
        })
        .sum();

    sum / actual.len() as f32
}

/// Performance metrics for inference
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub mean_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub std_dev_ms: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from a sorted vector of latencies (in milliseconds)
    pub fn from_sorted_latencies(mut latencies: Vec<f64>) -> Self {
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = latencies.len();
        let mean = latencies.iter().sum::<f64>() / n as f64;

        // Calculate standard deviation
        let variance = latencies
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        Self {
            mean_latency_ms: mean,
            p50_latency_ms: latencies[n / 2],
            p95_latency_ms: latencies[n * 95 / 100],
            p99_latency_ms: latencies[n * 99 / 100],
            min_latency_ms: latencies[0],
            max_latency_ms: latencies[n - 1],
            std_dev_ms: std_dev,
        }
    }

    /// Print performance metrics in a readable format
    pub fn print(&self) {
        println!("Performance Metrics:");
        println!("  Mean:   {:.2} ms", self.mean_latency_ms);
        println!("  P50:    {:.2} ms", self.p50_latency_ms);
        println!("  P95:    {:.2} ms", self.p95_latency_ms);
        println!("  P99:    {:.2} ms", self.p99_latency_ms);
        println!("  Min:    {:.2} ms", self.min_latency_ms);
        println!("  Max:    {:.2} ms", self.max_latency_ms);
        println!("  StdDev: {:.2} ms", self.std_dev_ms);
    }

    /// Assert that performance meets target criteria
    pub fn assert_meets_target(&self, target_p50_ms: f64, target_p95_ms: f64) {
        assert!(
            self.p50_latency_ms < target_p50_ms,
            "P50 latency {:.2}ms exceeds target {:.2}ms",
            self.p50_latency_ms,
            target_p50_ms
        );
        assert!(
            self.p95_latency_ms < target_p95_ms,
            "P95 latency {:.2}ms exceeds target {:.2}ms",
            self.p95_latency_ms,
            target_p95_ms
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_equal() {
        assert!(approx_equal(1.0, 1.00001, 1e-4));
        assert!(!approx_equal(1.0, 1.01, 1e-4));
    }

    #[test]
    fn test_compare_arrays_success() {
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![1.00001, 2.00001, 3.00001, 4.00001];

        assert!(compare_f32_arrays(&actual, &expected, 1e-4).is_ok());
    }

    #[test]
    fn test_compare_arrays_failure() {
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![1.0, 2.0, 3.1, 4.0]; // Larger difference

        assert!(compare_f32_arrays(&actual, &expected, 1e-5).is_err());
    }

    #[test]
    fn test_mean_absolute_error() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.1, 2.1, 3.1];

        let mae = mean_absolute_error(&actual, &expected);
        assert!((mae - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_performance_metrics() {
        let latencies = vec![10.0, 12.0, 11.0, 15.0, 13.0, 14.0, 20.0, 16.0, 17.0, 18.0];
        let metrics = PerformanceMetrics::from_sorted_latencies(latencies);

        assert!(metrics.mean_latency_ms > 0.0);
        assert!(metrics.p50_latency_ms <= metrics.p95_latency_ms);
        assert!(metrics.p95_latency_ms <= metrics.p99_latency_ms);
        assert!(metrics.min_latency_ms <= metrics.max_latency_ms);
    }
}
