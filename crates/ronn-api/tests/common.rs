//! Common test utilities and helpers for ronn-api tests

use std::path::PathBuf;

/// Get the path to the test fixtures directory
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Get the path to a specific test fixture file
pub fn fixture_path(filename: &str) -> PathBuf {
    fixtures_dir().join(filename)
}

/// Check if a test fixture exists
pub fn fixture_exists(filename: &str) -> bool {
    fixture_path(filename).exists()
}

/// Helper macro to conditionally run tests that require fixtures
#[macro_export]
macro_rules! require_fixture {
    ($fixture:expr) => {
        if !$crate::common::fixture_exists($fixture) {
            eprintln!(
                "Skipping test: fixture '{}' not found. See tests/fixtures/README.md",
                $fixture
            );
            return;
        }
    };
}

/// Helper to create a temporary invalid ONNX file for testing error paths
pub fn create_invalid_onnx_bytes() -> Vec<u8> {
    // Not a valid ONNX file - just random bytes
    vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA]
}

/// Helper to create an empty bytes vector
pub fn create_empty_bytes() -> Vec<u8> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixtures_dir_exists() {
        let dir = fixtures_dir();
        // Directory might not exist yet, but path should be constructible
        assert!(dir.to_string_lossy().contains("fixtures"));
    }

    #[test]
    fn test_invalid_onnx_bytes_not_empty() {
        let bytes = create_invalid_onnx_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_empty_bytes_is_empty() {
        let bytes = create_empty_bytes();
        assert!(bytes.is_empty());
    }
}
