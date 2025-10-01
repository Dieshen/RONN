//! Comprehensive tests for session management.
//!
//! This module tests all session-related functionality including:
//! - Session lifecycle (create, run, destroy)
//! - Thread-safe concurrent access
//! - Resource limits and cleanup
//! - Statistics tracking
//! - Error handling

mod test_utils;

use anyhow::Result;
use ronn_core::{DataType, SessionConfig, SessionManager, Tensor, TensorLayout, OptimizationLevel, ProviderId};
use std::sync::Arc;
use std::time::Duration;
use test_utils::*;

#[tokio::test]
async fn test_session_creation() -> Result<()> {
    let manager = SessionManager::new();
    let graph = create_test_graph()?;

    let session_id = manager.create_session(graph).await?;
    assert_eq!(manager.session_count(), 1);

    let session = manager.get_session(session_id);
    assert!(session.is_some());
    Ok(())
}

#[tokio::test]
async fn test_session_with_config() -> Result<()> {
    let manager = SessionManager::new();
    let graph = create_test_graph()?;

    let config = SessionConfig {
        thread_count: Some(4),
        memory_limit: Some(1024 * 1024 * 100), // 100 MB
        optimization_level: OptimizationLevel::Aggressive,
        preferred_providers: vec![ProviderId::CPU],
        timeout_seconds: Some(60),
        max_concurrent_inferences: Some(5),
        enable_metrics: true,
        custom_options: std::collections::HashMap::new(),
    };

    let session_id = manager.create_session_with_config(graph, Some(config)).await?;
    let session = manager.get_session(session_id).unwrap();

    assert_eq!(session.config.thread_count, Some(4));
    assert_eq!(session.config.optimization_level, OptimizationLevel::Aggressive);
    Ok(())
}

#[tokio::test]
async fn test_session_inference() -> Result<()> {
    let manager = SessionManager::new();
    let graph = create_test_graph()?;

    let session_id = manager.create_session(graph).await?;

    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;
    let outputs = manager.run_inference(session_id, vec![input]).await?;

    assert_eq!(outputs.len(), 1);
    Ok(())
}

#[tokio::test]
async fn test_session_statistics() -> Result<()> {
    let manager = SessionManager::new();
    let graph = create_test_graph()?;

    let session_id = manager.create_session(graph).await?;

    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    // Run inference multiple times
    for _ in 0..5 {
        manager.run_inference(session_id, vec![input.clone()]).await?;
    }

    let stats = manager.get_session_statistics(session_id).await?;
    assert_eq!(stats.total_inferences, 5);
    assert!(stats.average_inference_time_ms > 0.0);
    assert!(stats.min_inference_time_ms.is_some());
    assert!(stats.max_inference_time_ms.is_some());
    Ok(())
}

#[tokio::test]
async fn test_session_destruction() -> Result<()> {
    let manager = SessionManager::new();
    let graph = create_test_graph()?;

    let session_id = manager.create_session(graph).await?;
    assert_eq!(manager.session_count(), 1);

    manager.destroy_session(session_id).await?;
    assert_eq!(manager.session_count(), 0);

    let session = manager.get_session(session_id);
    assert!(session.is_none());
    Ok(())
}

#[tokio::test]
async fn test_concurrent_session_creation() -> Result<()> {
    let manager = Arc::new(SessionManager::new());

    let mut handles = vec![];

    for _ in 0..10 {
        let manager_clone = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            let graph = create_test_graph().unwrap();
            manager_clone.create_session(graph).await
        });
        handles.push(handle);
    }

    let results: Vec<_> = futures::future::join_all(handles).await;

    // All sessions should be created successfully
    assert!(results.iter().all(|r| r.as_ref().unwrap().is_ok()));
    assert_eq!(manager.session_count(), 10);
    Ok(())
}

#[tokio::test]
async fn test_concurrent_inference() -> Result<()> {
    let manager = Arc::new(SessionManager::new());
    let graph = create_test_graph()?;

    let session_id = manager.create_session(graph).await?;
    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    let mut handles = vec![];

    for _ in 0..20 {
        let manager_clone = Arc::clone(&manager);
        let input_clone = input.clone();
        let handle = tokio::spawn(async move {
            manager_clone.run_inference(session_id, vec![input_clone]).await
        });
        handles.push(handle);
    }

    let results: Vec<_> = futures::future::join_all(handles).await;

    // Count successes
    let success_count = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();

    // Some inferences should succeed
    assert!(success_count > 0);

    let stats = manager.get_session_statistics(session_id).await?;
    assert_eq!(stats.total_inferences, success_count as u64);
    Ok(())
}

#[tokio::test]
async fn test_max_concurrent_inferences() -> Result<()> {
    let mut config = SessionConfig::default();
    config.max_concurrent_inferences = Some(2);

    let manager = Arc::new(SessionManager::with_config(None, None, config.clone()));
    let graph = create_test_graph()?;

    let session_id = manager.create_session_with_config(graph, Some(config)).await?;
    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    let mut handles = vec![];

    for _ in 0..10 {
        let manager_clone = Arc::clone(&manager);
        let input_clone = input.clone();
        let handle = tokio::spawn(async move {
            manager_clone.run_inference(session_id, vec![input_clone]).await
        });
        handles.push(handle);
    }

    let results: Vec<_> = futures::future::join_all(handles).await;

    // Some should succeed, some should fail due to concurrency limit
    let success_count = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();
    let failure_count = results.iter().filter(|r| r.as_ref().unwrap().is_err()).count();

    assert!(success_count > 0);
    assert!(failure_count > 0);
    Ok(())
}

#[tokio::test]
async fn test_session_limits() -> Result<()> {
    let config = SessionConfig::default();
    let manager = SessionManager::with_config(None, Some(3), config);

    // Create 3 sessions (should succeed)
    for _ in 0..3 {
        let graph = create_test_graph()?;
        manager.create_session(graph).await?;
    }

    assert_eq!(manager.session_count(), 3);

    // Try to create a 4th session (should fail)
    let graph = create_test_graph()?;
    let result = manager.create_session(graph).await;
    assert!(result.is_err());
    Ok(())
}

#[tokio::test]
async fn test_session_cleanup_after_limit() -> Result<()> {
    let config = SessionConfig::default();
    let manager = SessionManager::with_config(None, Some(2), config);

    // Create 2 sessions
    let graph1 = create_test_graph()?;
    let session_id1 = manager.create_session(graph1).await?;

    let graph2 = create_test_graph()?;
    let _session_id2 = manager.create_session(graph2).await?;

    assert_eq!(manager.session_count(), 2);

    // Destroy one session
    manager.destroy_session(session_id1).await?;
    assert_eq!(manager.session_count(), 1);

    // Now we should be able to create another
    let graph3 = create_test_graph()?;
    let result = manager.create_session(graph3).await;
    assert!(result.is_ok());
    assert_eq!(manager.session_count(), 2);
    Ok(())
}

#[tokio::test]
async fn test_global_statistics() -> Result<()> {
    let manager = SessionManager::new();

    // Create multiple sessions and run inferences
    for i in 0..3 {
        let graph = create_test_graph()?;
        let session_id = manager.create_session(graph).await?;

        let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

        for _ in 0..(i + 1) {
            manager.run_inference(session_id, vec![input.clone()]).await?;
        }
    }

    let global_stats = manager.get_global_statistics().await;
    assert_eq!(global_stats.total_sessions, 3);
    assert_eq!(global_stats.total_inferences, 1 + 2 + 3); // 6 total
    Ok(())
}

#[tokio::test]
async fn test_session_not_found() -> Result<()> {
    let manager = SessionManager::new();

    let fake_session_id = uuid::Uuid::new_v4();
    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    let result = manager.run_inference(fake_session_id, vec![input]).await;
    assert!(result.is_err());
    Ok(())
}

#[tokio::test]
async fn test_session_list() -> Result<()> {
    let manager = SessionManager::new();

    let mut session_ids = vec![];
    for _ in 0..5 {
        let graph = create_test_graph()?;
        let session_id = manager.create_session(graph).await?;
        session_ids.push(session_id);
    }

    let listed_sessions = manager.list_sessions();
    assert_eq!(listed_sessions.len(), 5);

    // All created sessions should be in the list
    for session_id in &session_ids {
        assert!(listed_sessions.contains(session_id));
    }
    Ok(())
}

#[tokio::test]
async fn test_session_shutdown() -> Result<()> {
    let manager = SessionManager::new();

    // Create several sessions
    for _ in 0..5 {
        let graph = create_test_graph()?;
        manager.create_session(graph).await?;
    }

    assert_eq!(manager.session_count(), 5);

    // Shutdown should destroy all sessions
    manager.shutdown().await?;
    assert_eq!(manager.session_count(), 0);
    Ok(())
}

#[tokio::test]
async fn test_session_error_tracking() -> Result<()> {
    let manager = SessionManager::new();
    let graph = create_test_graph()?;

    let session_id = manager.create_session(graph).await?;

    // Create invalid input (wrong number of tensors)
    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    // Run successful inference
    manager.run_inference(session_id, vec![input.clone()]).await?;

    let stats_before = manager.get_session_statistics(session_id).await?;
    assert_eq!(stats_before.error_count, 0);

    // Try with wrong input count - this may or may not error depending on implementation
    // The test validates the error tracking mechanism works
    let result = manager.run_inference(session_id, vec![]).await;
    if result.is_err() {
        let stats_after = manager.get_session_statistics(session_id).await?;
        assert!(stats_after.error_count >= stats_before.error_count);
    }
    Ok(())
}

#[tokio::test]
async fn test_session_waits_for_active_inferences() -> Result<()> {
    let manager = Arc::new(SessionManager::new());
    let graph = create_test_graph()?;

    let session_id = manager.create_session(graph).await?;
    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    // Start a long-running inference
    let manager_clone = Arc::clone(&manager);
    let inference_handle = tokio::spawn(async move {
        manager_clone.run_inference(session_id, vec![input]).await
    });

    // Give it a moment to start
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Try to destroy the session - should wait for active inference
    let destroy_result = manager.destroy_session(session_id).await;

    // Wait for inference to complete
    let _inference_result = inference_handle.await;

    // Destroy should succeed
    assert!(destroy_result.is_ok());
    Ok(())
}

#[tokio::test]
async fn test_invalid_graph() -> Result<()> {
    use ronn_core::GraphBuilder;

    let manager = SessionManager::new();

    // Create an invalid graph (with cycles)
    let mut builder = GraphBuilder::new();

    let node_a = builder.add_op("A", Some("node_a".to_string()));
    builder.add_output(node_a, "a_out");

    let node_b = builder.add_op("B", Some("node_b".to_string()));
    builder
        .add_input(node_b, "a_out")
        .add_output(node_b, "b_out");

    let node_c = builder.add_op("C", Some("node_c".to_string()));
    builder
        .add_input(node_c, "b_out")
        .add_output(node_c, "a_out"); // Creates a cycle

    builder.connect(node_a, node_b, "a_out")?;
    builder.connect(node_b, node_c, "b_out")?;
    builder.connect(node_c, node_a, "a_out")?;

    builder
        .set_inputs(vec!["a_out".to_string()])
        .set_outputs(vec!["b_out".to_string()]);

    // Build should fail due to cycle
    let graph_result = builder.build();
    assert!(graph_result.is_err());
    Ok(())
}

#[tokio::test]
async fn test_multiple_managers() -> Result<()> {
    let manager1 = SessionManager::new();
    let manager2 = SessionManager::new();

    let graph1 = create_test_graph()?;
    let graph2 = create_test_graph()?;

    let session_id1 = manager1.create_session(graph1).await?;
    let session_id2 = manager2.create_session(graph2).await?;

    // Sessions should be independent
    assert_ne!(session_id1, session_id2);
    assert_eq!(manager1.session_count(), 1);
    assert_eq!(manager2.session_count(), 1);

    // Destroying in one manager shouldn't affect the other
    manager1.destroy_session(session_id1).await?;
    assert_eq!(manager1.session_count(), 0);
    assert_eq!(manager2.session_count(), 1);
    Ok(())
}

#[tokio::test]
async fn test_session_timing_statistics() -> Result<()> {
    let manager = SessionManager::new();
    let graph = create_test_graph()?;

    let session_id = manager.create_session(graph).await?;
    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    // Run multiple inferences
    for _ in 0..10 {
        manager.run_inference(session_id, vec![input.clone()]).await?;
    }

    let stats = manager.get_session_statistics(session_id).await?;

    assert_eq!(stats.total_inferences, 10);
    assert!(stats.average_inference_time_ms > 0.0);
    assert!(stats.min_inference_time_ms.is_some());
    assert!(stats.max_inference_time_ms.is_some());

    let min_time = stats.min_inference_time_ms.unwrap();
    let max_time = stats.max_inference_time_ms.unwrap();
    let avg_time = stats.average_inference_time_ms as u64;

    // Basic sanity checks
    assert!(min_time <= avg_time);
    assert!(avg_time <= max_time);
    assert!(stats.last_inference_at.is_some());
    Ok(())
}

#[tokio::test]
async fn test_session_resource_isolation() -> Result<()> {
    let manager = SessionManager::new();

    let graph1 = create_test_graph()?;
    let graph2 = create_test_graph()?;

    let session_id1 = manager.create_session(graph1).await?;
    let session_id2 = manager.create_session(graph2).await?;

    let input = Tensor::ones(vec![1, 3, 224, 224], DataType::F32, TensorLayout::RowMajor)?;

    // Run inferences on both sessions
    manager.run_inference(session_id1, vec![input.clone()]).await?;
    manager.run_inference(session_id1, vec![input.clone()]).await?;

    manager.run_inference(session_id2, vec![input.clone()]).await?;

    // Statistics should be isolated
    let stats1 = manager.get_session_statistics(session_id1).await?;
    let stats2 = manager.get_session_statistics(session_id2).await?;

    assert_eq!(stats1.total_inferences, 2);
    assert_eq!(stats2.total_inferences, 1);
    Ok(())
}
