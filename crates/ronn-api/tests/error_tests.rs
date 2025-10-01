//! Unit tests for Error types and error handling
//!
//! Tests error construction, conversions, Display formatting,
//! and error chaining behavior.

use ronn_api::Error;
use std::io;

#[test]
fn test_model_load_error_display() {
    let error = Error::ModelLoadError("File not found".to_string());
    let display = format!("{}", error);

    assert!(display.contains("Failed to load model"));
    assert!(display.contains("File not found"));
}

#[test]
fn test_inference_error_display() {
    let error = Error::InferenceError("Operator not supported".to_string());
    let display = format!("{}", error);

    assert!(display.contains("Inference failed"));
    assert!(display.contains("Operator not supported"));
}

#[test]
fn test_invalid_input_error_display() {
    let error = Error::InvalidInput("Missing tensor 'input'".to_string());
    let display = format!("{}", error);

    assert!(display.contains("Invalid input"));
    assert!(display.contains("Missing tensor"));
}

#[test]
fn test_session_error_display() {
    let error = Error::SessionError("Failed to initialize".to_string());
    let display = format!("{}", error);

    assert!(display.contains("Session error"));
    assert!(display.contains("Failed to initialize"));
}

#[test]
fn test_provider_error_display() {
    let error = Error::ProviderError("GPU not available".to_string());
    let display = format!("{}", error);

    assert!(display.contains("Provider error"));
    assert!(display.contains("GPU not available"));
}

#[test]
fn test_error_debug_format() {
    let error = Error::ModelLoadError("test".to_string());
    let debug = format!("{:?}", error);

    assert!(debug.contains("ModelLoadError"));
}

#[test]
fn test_io_error_conversion() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
    let error: Error = io_err.into();

    let display = format!("{}", error);
    assert!(display.contains("IO error"));
}

#[test]
fn test_io_error_preserves_message() {
    let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
    let error: Error = io_err.into();

    let display = format!("{}", error);
    assert!(display.contains("access denied"));
}

#[test]
fn test_error_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<Error>();
    assert_sync::<Error>();
}

#[test]
fn test_error_variants_distinct() {
    let e1 = Error::ModelLoadError("test".to_string());
    let e2 = Error::InferenceError("test".to_string());

    // Different variants should have different Display output
    let d1 = format!("{}", e1);
    let d2 = format!("{}", e2);

    assert_ne!(d1, d2);
}

#[test]
fn test_model_load_error_context() {
    let error = Error::ModelLoadError("path/to/model.onnx: corrupted file".to_string());
    let display = format!("{}", error);

    assert!(display.contains("path/to/model.onnx"));
    assert!(display.contains("corrupted"));
}

#[test]
fn test_inference_error_context() {
    let error = Error::InferenceError("Node 'Conv_5': invalid padding".to_string());
    let display = format!("{}", error);

    assert!(display.contains("Conv_5"));
    assert!(display.contains("invalid padding"));
}

#[test]
fn test_invalid_input_error_with_tensor_name() {
    let error = Error::InvalidInput("Required input 'data' not provided".to_string());
    let display = format!("{}", error);

    assert!(display.contains("data"));
    assert!(display.contains("not provided"));
}

#[test]
fn test_provider_error_with_details() {
    let error = Error::ProviderError("CUDA provider unavailable: CUDA runtime not found".to_string());
    let display = format!("{}", error);

    assert!(display.contains("CUDA"));
    assert!(display.contains("runtime not found"));
}

#[test]
fn test_result_type_alias() {
    // Test that Result<T> works correctly
    fn returns_ok() -> ronn_api::Result<i32> {
        Ok(42)
    }

    fn returns_err() -> ronn_api::Result<i32> {
        Err(Error::InferenceError("test error".to_string()))
    }

    assert!(returns_ok().is_ok());
    assert!(returns_err().is_err());
}

#[test]
fn test_error_propagation() {
    fn inner() -> ronn_api::Result<()> {
        Err(Error::InvalidInput("bad input".to_string()))
    }

    fn outer() -> ronn_api::Result<()> {
        inner()?;
        Ok(())
    }

    let result = outer();
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(matches!(e, Error::InvalidInput(_)));
    }
}

#[test]
fn test_error_downcasting_preserves_type() {
    let error = Error::ModelLoadError("test".to_string());

    match error {
        Error::ModelLoadError(msg) => assert_eq!(msg, "test"),
        _ => panic!("Wrong error variant"),
    }
}

#[test]
fn test_multiple_error_types() {
    let errors = vec![
        Error::ModelLoadError("e1".to_string()),
        Error::InferenceError("e2".to_string()),
        Error::InvalidInput("e3".to_string()),
        Error::SessionError("e4".to_string()),
        Error::ProviderError("e5".to_string()),
    ];

    // All should be displayable
    for error in errors {
        let display = format!("{}", error);
        assert!(!display.is_empty());
    }
}

#[test]
fn test_error_can_be_boxed() {
    let error: Box<dyn std::error::Error> = Box::new(
        Error::InferenceError("test".to_string())
    );

    assert!(error.to_string().contains("Inference failed"));
}

#[test]
fn test_error_implements_error_trait() {
    fn accepts_error(_e: impl std::error::Error) {}

    accepts_error(Error::ModelLoadError("test".to_string()));
}

#[test]
fn test_io_error_kinds() {
    let test_cases = vec![
        (io::ErrorKind::NotFound, "not found"),
        (io::ErrorKind::PermissionDenied, "permission denied"),
        (io::ErrorKind::InvalidData, "invalid data"),
    ];

    for (kind, msg) in test_cases {
        let io_err = io::Error::new(kind, msg);
        let error: Error = io_err.into();
        let display = format!("{}", error);

        assert!(display.contains(msg));
    }
}

#[test]
fn test_error_messages_are_helpful() {
    let errors = vec![
        (
            Error::ModelLoadError("model.onnx not found".to_string()),
            vec!["model.onnx", "not found"]
        ),
        (
            Error::InvalidInput("Missing required input: data".to_string()),
            vec!["Missing", "input", "data"]
        ),
        (
            Error::ProviderError("GPU provider not available".to_string()),
            vec!["GPU", "provider", "not available"]
        ),
    ];

    for (error, keywords) in errors {
        let display = format!("{}", error);
        for keyword in keywords {
            assert!(
                display.contains(keyword),
                "Error message '{}' should contain '{}'",
                display,
                keyword
            );
        }
    }
}

#[test]
fn test_error_empty_message() {
    let error = Error::InferenceError(String::new());
    let display = format!("{}", error);

    // Should still have the error type prefix
    assert!(display.contains("Inference failed"));
}

#[test]
fn test_error_very_long_message() {
    let long_msg = "x".repeat(1000);
    let error = Error::ModelLoadError(long_msg.clone());
    let display = format!("{}", error);

    // Should contain the message (or at least part of it)
    assert!(display.len() > 100);
}
